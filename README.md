# Rinha de Backend 2026 — Python

Solução em Python para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

API de detecção de fraude em transações de cartão de crédito usando busca vetorial (K-NN com K=5) sobre 3 milhões de referências pré-rotuladas.

## Resultado na Prévia

| Métrica | Valor |
|---------|-------|
| **Score Final** | **4497** |
| Score p99 | 1931 (p99 = 11.7ms) |
| Score detecção | 2566 |
| False Positives | 6 |
| False Negatives | 7 |
| HTTP Errors | 0 |
| Failure Rate | 0.02% |

## Stack

- **Python 3.12** + uvicorn (ASGI raw, sem framework)
- **NumPy** para centroid search e fallback de busca vetorial
- **Extensão C com SSE4.1** para scan de clusters (hot path)
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
make help     # Ver todos os comandos
```

## Estrutura

```
src/
├── server.py          # App ASGI raw (endpoints /ready e /fraud-score)
├── vectorizer.py      # Payload JSON → vetor int16[14]
├── index.py           # Loader mmap + busca IVF (C + NumPy fallback)
├── scan.c             # Extensão C com SSE4.1 para scan de clusters
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
2. **Centroid search** — matmul 4096×14 para encontrar os 7 clusters mais próximos (NumPy)
3. **Cluster scan** — para cada cluster (~730 vetores), calcula distâncias e mantém top-5 (C com SSE4.1)
4. **Resposta** — fraud_count/5 = score, pre-computado em bytes

### Extensão C (scan.c)

O hot path (scan de clusters) é implementado em C com SIMD:
- Dot product de 14 dimensões usando SSE4.1 (`_mm_cvtepi16_epi32`, `_mm_mullo_epi32`)
- 12 dims processadas em 3 operações SIMD + 2 dims scalar
- Top-5 mantido incrementalmente (sem alocação)
- Fallback automático para NumPy se `scan.so` não existir

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
- **nprobe=7** — melhor score total (acurácia compensa latência extra)
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

### V6 — Extensão C com SIMD
- `scan.c` com SSE4.1 para dot product de 14 dims
- Top-5 mantido incrementalmente em C (sem NumPy no hot path)
- Fallback automático para NumPy se `scan.so` não existir
- **p99 na máquina oficial: 144ms → 11.7ms**
- **Score na prévia: 3405 → 4497**

## Decisões Descartadas

- **Brute force KNN** — O(N×14) por query em 3M vetores, inviável
- **FAISS/sklearn** — overhead de dependência e memória, não cabe em 170MB
- **K=8192 com nprobe=10** — estourou CPU no Mac Mini (HTTP errors)
- **K=32768** — K-means com muitos clusters perdeu acurácia (sample insuficiente)
- **Adaptive repair** — nprobe fixo=7 superou adaptive em todas as configurações testadas
- **2 uvicorn workers** — duplica memória, context switching piora p99 no Mac Mini
- **Fallback approved=true** — trocado para approved=false (FP custa 1 vs FN custa 3)

## Licença

MIT
