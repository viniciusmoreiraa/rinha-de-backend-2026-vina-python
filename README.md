# Rinha de Backend 2026 — Python

Solução em Python para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

API de detecção de fraude em transações de cartão de crédito usando busca vetorial (K-NN com K=5) sobre 3 milhões de referências pré-rotuladas.

## Stack

- **Python 3.12** + uvicorn (ASGI raw, sem framework)
- **NumPy** para busca vetorial (IVF com K-means)
- **orjson** para parsing JSON
- **uvloop** + **httptools** para I/O
- **nginx** como load balancer (round-robin)

## Arquitetura

```
Port 9999 → [nginx] → round-robin → [api1:8080]
                                   → [api2:8080]
                                        ↓
                              [index.bin via mmap]
```

| Serviço | CPU  | RAM   |
|---------|------|-------|
| nginx   | 0.10 | 10MB  |
| api1    | 0.45 | 170MB |
| api2    | 0.45 | 170MB |
| **Total** | **1.00** | **350MB** |

O índice IVF é construído offline durante `docker build` e carregado via `mmap` com páginas compartilhadas entre as duas instâncias.

## Como rodar

```bash
# Build (inclui construção do índice ~2-5min)
make build

# Subir
make up

# Verificar readiness
make ready

# Smoke test
make smoke

# Benchmark
make bench
```

## Comandos úteis

```bash
make stats        # Monitorar CPU/memória
make test-one     # Um POST manual
make bench-heavy  # Benchmark com 1000 requests
make restart      # Rebuild + restart
make down         # Derrubar
make clean        # Limpar tudo
```

## Estrutura

```
src/
├── server.py          # App ASGI (endpoints /ready e /fraud-score)
├── vectorizer.py      # Payload JSON → vetor int16[14]
├── index.py           # Loader mmap + busca IVF
└── build_index.py     # Constrói index.bin offline (K-means)
scripts/
├── smoke_test.py      # Validação do contrato da API
└── bench.py           # Benchmark de latência
```

## Estratégia

1. **Índice IVF offline** — K-means (K=1024) sobre os 3M vetores de referência, construído durante `docker build`
2. **Quantização int16** — vetores armazenados com escala 10000, metade da memória vs float32
3. **Busca adaptativa** — probe inicial em 2 clusters (~6K vetores); se resultado borderline, expande busca
4. **Respostas pre-computadas** — só 6 resultados possíveis (fraud_count 0-5)
5. **Zero HTTP errors** — qualquer exceção retorna fallback válido (custo de erro HTTP = 5 pontos)

## Licença

MIT
