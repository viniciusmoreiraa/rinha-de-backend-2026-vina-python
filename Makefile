.PHONY: build up down restart logs stats ready smoke bench clean k6 k6-smoke test help push

# Ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  make build       Build da imagem Docker (inclui index offline)"
	@echo "  make up          Subir containers"
	@echo "  make down        Derrubar containers"
	@echo "  make restart     Rebuild + restart"
	@echo "  make test        Rebuild + restart + k6 completo"
	@echo "  make k6          Teste k6 oficial (sem rebuild)"
	@echo "  make k6-smoke    Smoke test k6 oficial"
	@echo "  make smoke       Smoke test Python (50 payloads)"
	@echo "  make bench       Benchmark leve (200 req)"
	@echo "  make bench-heavy Benchmark pesado (1000 req)"
	@echo "  make test-one    Um POST manual"
	@echo "  make ready       Verificar /ready"
	@echo "  make stats       Monitorar CPU/memória"
	@echo "  make logs        Logs dos serviços"
	@echo "  make push        Build e push da imagem para Docker Hub"
	@echo "  make clean       Limpar containers e imagens"

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

# Rebuild + restart + teste oficial k6
test: down build up
	@sleep 5
	cd ../rinha-de-backend-2026/test && k6 run test.js && cat test/results.json | python3 -m json.tool

# Teste oficial k6 (sem rebuild)
k6:
	cd ../rinha-de-backend-2026/test && k6 run test.js && cat test/results.json | python3 -m json.tool

# Smoke test oficial k6
k6-smoke:
	cd ../rinha-de-backend-2026/test && k6 run smoke.js

# Build e push da imagem para Docker Hub
push:
	docker build --platform linux/amd64 -t vinimoreira/rinha-2026-python:latest .
	docker push vinimoreira/rinha-2026-python:latest

# Limpar containers e imagens
clean: down
	docker compose rm -f
	docker rmi -f $$(docker compose config --images 2>/dev/null) 2>/dev/null || true
