run:
	cargo build && RUST_BACKTRACE=1 ./target/debug/rust-nn ./assets/mnist_2.png