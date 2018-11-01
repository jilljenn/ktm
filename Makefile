DATASETS=$(wildcard data/*/data.csv)
FIRST_FEATURES=$(DATASETS:data.csv=X-ui.npz)
FEATURES=$(wildcard data/dummy/X-*.npz)
RESULTS=$(FEATURES:npz=lr.txt) $(FEATURES:npz=fm.txt)
BIG_FEATURES=$(wildcard data/assistments09/X-*.npz)
BIG_RESULTS=$(BIG_FEATURES:npz=lr.txt) $(BIG_FEATURES:npz=fm.txt)

all: $(FIRST_FEATURES) $(RESULTS)
	tail -n 10 data/dummy/*.txt

big: $(BIG_RESULTS)
	tail -n 10 data/assistments09/*.txt

data/%/X-ui.npz: data/%/data.csv
	python encode.py --dataset $* --users --items
	python encode.py --dataset $* --skills --wins --fails
	python encode.py --dataset $* --items --skills --wins --fails

X-%.lr.txt: X-%.npz
	time python lr.py $< > $@

X-%.fm.txt: X-%.npz
	time python fm.py $< > $@

clean:
	rm -f data/dummy/*.npz data/dummy/*.npy data/dummy/*.txt

fullclean:
	rm -f data/*/*.npz data/*/*.npy data/*/*.txt

tmp:
	pandoc README.md -o README.html

movie:
	python fm.py --d 20 data/movie100k/X-ui.npz
