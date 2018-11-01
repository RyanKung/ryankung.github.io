defualt:
	make build
	make ipfs

ipfs:
	cp *.html depoly
	cp -r css depoly
	cp -r images depoly
	cp -r pdfs depoly
	cp -r posts depoly
	ipfs add -r depoly

build:
	cd src; make
