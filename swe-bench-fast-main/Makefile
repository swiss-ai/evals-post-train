.PHONY: build clean test vet fmt

BINARY := swe-bench-fast
DIST := dist
MODULE := github.com/greynewell/swe-bench-fast
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS := -ldflags "-X main.version=$(VERSION)"

build:
	go build $(LDFLAGS) -o $(DIST)/$(BINARY) ./cmd/swe-bench-fast

clean:
	rm -rf $(DIST)

test:
	go test -race ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

lint: vet fmt

all: clean build test
