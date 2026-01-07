SOURCES=$(wildcard src/*.rs)


all: $(SOURCES) Makefile
	cargo build

rebuild:
	make clean
	make all

commit:
	aic -ac
	git push

test:
	LLAMATEST_GGUF=qwen2.5-0.5b-instruct-q3_k_m.gguf cargo test -- --show-output

clean:
	cargo clean
	cargo update

all:
