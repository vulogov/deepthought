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
	cargo test -- --show-output

clean:
	cargo clean
	cargo update

all:
