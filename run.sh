#!/bin/bash


for ((i=0; i<10; i+=1))
do
	docker compose up
	docker compose down
done
