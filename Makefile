.PHONY: build up down restart logs stats ready smoke bench clean

# Build da imagem Docker (inclui build do index offline)
build:
	docker compose build

# Subir todos os serviços
up:
	docker compose up -d

# Derrubar todos os serviços
down:
	docker compose down

# Rebuild + restart
restart: down build up

# Logs dos serviços
logs:
	docker compose logs -f

# Monitorar CPU/memória
stats:
	docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Verificar readiness
ready:
	@curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:9999/ready

# Smoke test (valida contrato da API)
smoke:
	python3 scripts/smoke_test.py http://localhost:9999

# Benchmark de latência (200 requests, concurrency 4)
bench:
	python3 scripts/bench.py http://localhost:9999 200 4

# Benchmark pesado (1000 requests, concurrency 10)
bench-heavy:
	python3 scripts/bench.py http://localhost:9999 1000 10

# Teste rápido: um POST manual
test-one:
	@curl -s -X POST http://localhost:9999/fraud-score \
		-H "Content-Type: application/json" \
		-d '{"id":"tx-test","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.71},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.86}}' | python3 -m json.tool

# Limpar containers e imagens
clean: down
	docker compose rm -f
	docker rmi -f $$(docker compose config --images 2>/dev/null) 2>/dev/null || true
