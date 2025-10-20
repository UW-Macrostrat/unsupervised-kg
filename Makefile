# Makefile for Docker-based workflow

CURRENT_DIR := $(shell pwd)
CONTAINER_NAME := unsupervised_kg
IMAGE_NAME := unsupervised_kg:latest

# Default target
.PHONY: docker
docker: build run exec

# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME) docker/.

# Run the Docker container in detached mode
.PHONY: run
run:
	docker run --gpus all -d -v $(CURRENT_DIR):/working_dir/ --name=$(CONTAINER_NAME) $(IMAGE_NAME) sleep infinity

# Open a bash shell in the running container
.PHONY: exec
exec:
	docker exec -it $(CONTAINER_NAME) bash

# Stop and remove the container
.PHONY: clean
clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Full reset: stop, remove, and rebuild
.PHONY: reset
reset: clean build run exec
