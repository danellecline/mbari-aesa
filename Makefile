.PHONY: all compose clean stop tag_latest push_latest

all: compose

compose:
	docker-compose up -d

clean:
	docker-compose rm --force

stop:
	docker-compose stop

tag_latest:
	pushd worker; make build tag_latest; popd
	pushd submit; make build tag_latest; popd

push_latest:
	pushd worker; make build push push_latest; popd
	pushd submit; make build push push_latest; popd

push:
	pushd worker; make build push push; popd
	pushd submit; make build push push; popd