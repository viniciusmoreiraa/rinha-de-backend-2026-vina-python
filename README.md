# Rinha de Backend 2026 — Python

Solução em Python para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

API de detecção de fraude em transações de cartão de crédito usando busca vetorial (K-NN com K=5) sobre 3 milhões de referências pré-rotuladas.

## Resultado nas Prévias

| Prévia | Score | p99 | FP | FN | Err |
|--------|-------|-----|----|----|-----|
| 1 | 4947 | 4.3ms | 6 | 6 | 0 |
| **2** | **4950** | **4.0ms** | **6** | **8** | **0** |

## Stack

- **Python 3.12** + uvicorn (ASGI raw, sem framework)
- **NumPy** para busca vetorial (IVF com K-means)
- **orjson** para parsing JSON
- **uvloop** + **httptools** para I/O
- **nginx** como load balancer (round-robin)

## Arquitetura

```
Port 9999 → [nginx LB] → round-robin → [api1:8080]
                                      → [api2:8080]
                                           ↓
                                 [index.bin via mmap]
```

| Serviço | CPU  | RAM   |
|---------|------|-------|
| nginx   | 0.16 | 10MB  |
| api1    | 0.42 | 170MB |
| api2    | 0.42 | 170MB |
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
├── index.py           # Loader mmap + busca IVF
└── build_index.py     # Constrói index.bin offline (K-means)
scripts/
├── smoke_test.py      # Validação do contrato da API
├── bench.py           # Benchmark de latência
├── bench_nprobe.py    # Comparação de nprobe values
├── accuracy_test.py   # Teste de acurácia vs brute force
└── stress.js          # Stress test k6 (1500 req/s)
```

## Estratégia Técnica

### Índice IVF offline

O `build_index.py` roda durante `docker build` e produz `index.bin`:
- Parse de `references.json.gz` (3M vetores de 14 dimensões)
- K-means com **K=4096 clusters** (MiniBatchKMeans, 80K samples)
- Quantização dos vetores para **int16** (escala 10000) — metade da memória vs float32
- Pré-computação de normas quadradas (`vector_sq`) em int64
- Bounding boxes por cluster para poda
- Serialização em formato binário customizado (~95MB)

### Busca vetorial

Cada request faz:
1. **Vectorize** — payload JSON → vetor int16[14] (Python puro, sem NumPy)
2. **Centroid search** — matmul 4096×14 para encontrar os 5 clusters mais próximos (NumPy)
3. **Cluster scan** — para cada cluster (~730 vetores), calcula distâncias e mantém top-5 (NumPy otimizado)
4. **Adaptive repair** — se fraud_count é 2 ou 3 (borderline), escaneia até 3 clusters adicionais com poda por bounding box
5. **Resposta** — fraud_count/5 = score, pre-computado em bytes

### Busca adaptativa (nprobe=5 + repair)

A estratégia combina velocidade com acurácia:
- **Caso comum (~97%)**: fraud_count é 0, 1, 4 ou 5 → retorna após 5 clusters (rápido)
- **Caso borderline (~3%)**: fraud_count é 2 ou 3 → escaneia até 3 clusters extras com poda por bbox
- Resultado: latência baixa na maioria dos requests, acurácia alta nos casos difíceis

### Respostas pre-computadas

Só existem 6 respostas possíveis (fraud_count 0-5). Os dicts ASGI (headers + body) são criados no startup e reutilizados — zero alocação por request na camada HTTP.

## Evolução da Solução

### V1 — Baseline funcional
- IVF com K=1024, nprobe=2
- NumPy para tudo (scan, merge, distâncias)
- Mediana ~11ms, p99 ~88ms (local)

### V2 — Otimizações NumPy
- **Inner loop vetorizado** — substituiu loop Python por `np.argpartition` + `np.concatenate`
- **Dot product expandido** — evita array intermediário `diff`: `dist = ||a||² + ||b||² - 2*a·b`
- **Pre-reduce no merge** — limita candidatos a 5 antes de mergear
- **Buffers reutilizáveis** — `_query_i32`, `_top5_dists`, `_merge_dists` pré-alocados no `__init__`
- Mediana ~4ms, p99 ~64ms (local)

### V3 — Tuning de K e nprobe
- Testamos K=1024, 2048, 4096, 8192, 32768
- **K=4096** — melhor equilíbrio (clusters de ~730 vetores)
- Testamos nprobe=1 a 15 com k6 oficial
- Score local: ~4800

### V4 — Vectorizer sem NumPy
- Todas as operações do vectorize em Python puro (sem `np.array`, sem `np.clip`)
- Lookup tables pre-computadas: `_HOUR_Q[24]`, `_WEEKDAY_Q[7]`, `MCC_RISK_Q`
- Buffer `_OUT_BUF` reutilizado entre requests
- Timestamp parsing otimizado para 2026 (constante `_DAYS_2026`)

### V5 — Otimizações de latência
- **`np.multiply(dot, 2, dtype=np.int64)`** — evita `dot.astype(int64)` temporário
- **`vector_sq` carregado como int64** no `__init__` — evita cast por request
- **`q_sq = int(q @ q)`** — dot product sem array intermediário
- **`argsort` no merge** — mais rápido que `argpartition` para ≤10 elementos
- **Scan inlined** no loop do `search()` — elimina overhead de chamada de método
- **nginx tuning** — `tcp_nodelay`, `proxy_buffering off`, `keepalive 128`, `backlog 2048`
- **Eventos ASGI pre-computados** — `STARTS[]`, `BODY_EVENTS[]` criados no startup

### V6 — Busca adaptativa
- **nprobe=5 + adaptive repair em {2,3}** — superou nprobe=7 fixo
- 97% dos requests escaneiam apenas 5 clusters (vs 7 antes)
- 3% borderline escaneiam até 8 clusters com poda por bbox
- **p99 na máquina oficial: 11.7ms → 4.3ms**
- **Score na prévia: 4497 → 4947**

## Decisões Descartadas

- **Brute force KNN** — O(N×14) por query em 3M vetores, inviável
- **FAISS/sklearn** — overhead de dependência e memória, não cabe em 170MB
- **K=8192 com nprobe=10** — estourou CPU no Mac Mini (HTTP errors)
- **K=32768** — K-means com muitos clusters perdeu acurácia (sample insuficiente)
- **nprobe=7 fixo** — superado por nprobe=5 + adaptive repair{2,3}
- **Extensão C (scan.c)** — testada com SSE4.1 e scalar, mas NumPy otimizado foi mais estável e suficientemente rápido
- **2 uvicorn workers** — duplica memória, context switching piora p99 no Mac Mini
- **Fallback approved=true** — trocado para approved=false (FP custa 1 vs FN custa 3)

## Licença

MIT
