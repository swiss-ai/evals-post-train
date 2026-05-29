FROM golang:1.24-alpine AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags "-s -w" -o /swe-bench-fast ./cmd/swe-bench-fast

FROM scratch
COPY --from=builder /swe-bench-fast /swe-bench-fast
ENTRYPOINT ["/swe-bench-fast"]
