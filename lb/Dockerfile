FROM --platform=linux/amd64 gcc:14 AS builder
WORKDIR /build
COPY main.c .
RUN gcc -O3 -static -o /lb main.c

FROM --platform=linux/amd64 scratch
COPY --from=builder /lb /lb
CMD ["/lb"]
