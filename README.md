# Rinha de Backend 2026 — Python

Solução em Python para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

API de detecção de fraude em transações de cartão de crédito usando busca vetorial (K-NN com K=5) sobre 3 milhões de referências pré-rotuladas.

## Resultado

| Métrica | Valor |
|---------|-------|
| **Score** | **5871** |
| p99 | 1.34ms |
| FP | 0 |
| FN | 0 |
| Detection Score | 3000 (máximo) |

## Stack

- **Python 3.12** + uvicorn (ASGI raw, sem framework)
- **NumPy** para busca vetorial (IVF com K-means)
- **orjson** para parsing JSON
- **uvloop** + **httptools** para I/O
- **Load balancer em C** com `splice()` (zero-copy)

## Arquitetura

```
Port 9999 → [C LB splice] → round-robin → [api1 UDS]
                                         → [api2 UDS]
                                              ↓
                                    [index.bin via mmap]
```

| Serviço | CPU  | RAM   |
|---------|------|-------|
| LB      | 0.16 | 30MB  |
| api1    | 0.42 | 160MB |
| api2    | 0.42 | 160MB |
| **Total** | **1.00** | **350MB** |

## Como rodar

```bash
make build    # Build da imagem (inclui index offline ~30s)
make up       # Subir containers
make ready    # Verificar /ready
make test     # Rebuild + restart + k6 completo
make push     # Build e push para Docker Hub
make help     # Ver todos os comandos
```

## Estrutura

```
src/
├── server.py          # App ASGI raw (endpoints /ready e /fraud-score)
├── vectorizer.py      # Payload JSON → vetor int16[14]
├── index.py           # Loader mmap + busca IVF adaptativa
├── build_index.py     # Constrói index.bin offline (K-means)
├── config.py          # Configuração
scripts/
├── smoke_test.py      # Validação do contrato da API
├── bench.py           # Benchmark de latência
├── bench_nprobe.py    # Comparação de nprobe values
├── accuracy_test.py   # Teste de acurácia vs brute force
└── stress.js          # Stress test k6 (1500 req/s)
```

---

## Estratégia Técnica

### Índice IVF offline

O `build_index.py` roda durante `docker build` e produz `index.bin`:
- Parse de `references.json.gz` (3M vetores de 14 dimensões)
- K-means com **K=4096 clusters** (MiniBatchKMeans, 80K samples)
- Quantização dos vetores para **int16** (escala 10000) — metade da memória vs float32
- Pré-computação de normas quadradas (`vector_sq`) em int64
- Bounding boxes por cluster para poda na fase de repair
- Serialização em formato binário customizado (~95MB)

### Busca vetorial — o hot path

Cada request faz:
1. **Vectorize** — payload JSON → vetor int16[14] (Python puro, sem NumPy)
2. **Centroid search** — matmul 4096×14 para encontrar os N clusters mais próximos
3. **Cluster scan** — para cada cluster (~730 vetores), calcula distâncias euclidiana e mantém top-5
4. **Adaptive repair** — se o resultado é borderline, busca clusters adicionais filtrados por bounding box
5. **Resposta** — fraud_count/5 = score, indexa resposta pre-computada

### Busca adaptativa: a decisão mais importante

Esta é a otimização que mais impactou o resultado. A ideia:

**Fase 1 — Probe inicial (rápido):** busca nos `nprobe` clusters mais próximos (configurável, atualmente 3). Cobre ~97% dos casos com confiança.

**Decisão:** se `fraud_count` é 0 ou 5 → resultado confiável, retorna direto. Se está entre `REPAIR_MIN` e `REPAIR_MAX` (1-4) → resultado ambíguo, entra na fase de repair.

**Fase 2 — Repair (só borderline ~3-4%):** filtra TODOS os 4096 clusters por bounding box distance numa única operação NumPy vetorizada. Só escaneia os poucos clusters cujo bbox pode conter um vizinho mais próximo.

Resultado: latência quase idêntica ao probe-only para queries fáceis, e acurácia equivalente a nprobe=100+ para queries difíceis.

#### Por que funciona

Investigamos os erros de classificação e descobrimos que:
- Com nprobe=9 e sem repair, tínhamos 5 erros em 54100
- Mesmo com nprobe=20, 3 erros persistiam
- Só nprobe=100 eliminava todos — buscar 100 de 4096 clusters é lento
- O repair com bbox filter resolve isso: pré-filtra todos os 4096 clusters com uma operação vetorizada barata, depois escaneia apenas os que passam (~5-15 clusters)
- Com isso, nprobe=3 + repair = 0 erros, com latência próxima de nprobe=3 puro

#### Bug corrigido: `break` → `continue` → bbox batch

O código original do repair usava `break` ao encontrar um cluster cujo bounding box era longe demais. Isso parava a busca prematuramente porque os clusters são ordenados por distância ao centroide, não por distância do bounding box — um cluster com centroide mais distante pode ter bbox mais próximo.

Testamos `continue` (pular em vez de parar), que corrigiu a acurácia mas deixou o loop lento. A solução final: pré-computar TODAS as distâncias de bbox em uma única operação NumPy (4096×14 → 4096 distâncias), filtrar, e iterar só os sobreviventes. Zero loop Python desnecessário.

### Otimizações de runtime

**Vectorizer:**
- Função `_q()` (clamp + quantize) inlined — elimina 8 chamadas de função por request
- Constante Sakamoto pré-computada para 2026 — elimina 3 divisões inteiras
- Timestamp parseado uma vez e reutilizado para hora, weekday e minutos

**Index:**
- `offsets` convertidos para Python list no init — elimina conversão numpy→int no loop
- `worst_val` mantido como scalar Python — evita `top5_d.max()` repetido
- Buffers de bbox repair pré-alocados — zero alocação na fase de repair

**Server:**
- `gc.disable()` após carregar index — numpy usa refcount, sem circular refs
- Respostas ASGI pre-computadas — 6 possíveis, zero alocação por request
- `_read_body` com fast path para body em chunk único

**Dockerfile:**
- `PYTHONOPTIMIZE=2` — remove asserts e docstrings
- `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1` / `OMP_NUM_THREADS=1` — evita threads desnecessárias no numpy
- `MALLOC_ARENA_MAX=1` — reduz fragmentação com worker único

### Respostas pre-computadas

Só existem 6 respostas possíveis (fraud_count 0-5). Os dicts ASGI (headers + body) são criados no startup e reutilizados — zero alocação por request na camada HTTP.

---

## Evolução da Solução

### V1 — Baseline funcional
- IVF com K=1024, nprobe=2
- NumPy para tudo (scan, merge, distâncias)
- Mediana ~11ms, p99 ~88ms (local)

### V2 — Otimizações NumPy
- Inner loop vetorizado, dot product expandido (`dist = ||a||² + ||b||² - 2*a·b`)
- Buffers reutilizáveis pré-alocados no `__init__`
- Mediana ~4ms, p99 ~64ms (local)

### V3 — Tuning de K e nprobe
- Testamos K=1024, 2048, 4096, 8192, 32768
- **K=4096** melhor equilíbrio (clusters de ~730 vetores)

### V4 — Vectorizer sem NumPy
- Todas as operações em Python puro, lookup tables pre-computadas
- Buffer `_OUT_BUF` reutilizado entre requests

### V5 — Otimizações de latência
- `np.multiply(dot, 2, dtype=np.int64)` evita cast temporário
- Eventos ASGI pre-computados, LB em C com splice

### V6 — Busca adaptativa (nprobe=5 + repair)
- Superou nprobe=7 fixo: 97% dos requests escaneiam apenas 5 clusters
- Score prévia: 4947

### V7 — Repair com bbox batch + tuning agressivo
- Corrigido bug do `break` prematuro no repair
- Bbox filter vetorizado em todos os 4096 clusters
- nprobe reduzido de 9 → 3 com 0 erros (repair compensa)
- Vectorizer otimizado (inline _q, Sakamoto pré-computado)
- GC desabilitado, env vars de runtime
- **Score: 5871 (detection perfeita: 3000)**

---

## Decisões Descartadas

| Tentativa | Por que descartada |
|-----------|-------------------|
| Brute force KNN | O(N×14) por query em 3M vetores, inviável |
| FAISS/sklearn runtime | Overhead de dependência e memória |
| K=8192/32768 | Estourou CPU ou perdeu acurácia |
| nprobe alto fixo (9-20) | Lento, repair adaptativo é melhor |
| `dists_buf` pré-alocado | 2 ops numpy > 1 alocação — medimos e era mais lento |
| `argpartition` no merge | Não mais rápido que `argsort` para ≤10 elementos |
| Sort de clusters iniciais | Adicionava overhead em todas as queries |
| Extensão C (scan.c com SSE4.1) | NumPy otimizado foi suficiente |
| 2 uvicorn workers | Duplica memória, context switching piora p99 |

## Aprendizados

1. **Medir antes de otimizar.** Várias "otimizações" (dists_buf, argpartition no merge, sort de clusters) pioraram o p99. Numpy tem overhead fixo por chamada — reduzir alocações só vale se não aumentar o número de chamadas.

2. **Adaptive > brute force.** nprobe=3 + repair inteligente supera nprobe=100 em velocidade com a mesma acurácia. A chave é que ~97% dos casos são "fáceis" e não precisam de trabalho extra.

3. **Bbox filter vetorizado é barato.** Filtrar 4096 clusters com bounding box numa única operação NumPy (4096×14) custa microsegundos. Mais barato que um loop Python com 10 iterações.

4. **O `break` vs `continue` no repair importa.** Clusters ordenados por centroide não garante ordem de bounding box. O `break` original descartava vizinhos corretos. Mas `continue` sem limite era lento. A solução: pré-filtrar tudo de uma vez.

5. **Otimizações de runtime somam.** `gc.disable()`, `PYTHONOPTIMIZE=2`, thread pinning do numpy — individualmente pequenas, juntas ~15% no p99.

## Licença

MIT
