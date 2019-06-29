tuto:
	pandoc -s --bibliography=biblio.bib --biblatex tuto.md -t beamer -o tuto.tex
	pdflatex tuto
# 	biber tuto
# 	pdflatex tuto
	open tuto.pdf

all:
	pandoc -s --bibliography=biblio.bib --biblatex inria.md -t beamer -o inria.tex
	pdflatex inria.tex
	biber inria
	pdflatex inria
	open inria.pdf

hyderabad:
	pandoc -s --bibliography=biblio.bib --biblatex hyderabad.md -t beamer -o hyderabad.tex
	pdflatex hyderabad.tex
	biber hyderabad
	pdflatex hyderabad
	open hyderabad.pdf

shanghai:
	pandoc -s --bibliography=biblio.bib --biblatex shanghai.md -t beamer -o shanghai.tex
	# pdflatex shanghai.tex
	xelatex shanghai.tex
	biber shanghai
	xelatex shanghai
	# pdflatex shanghai
	open shanghai.pdf

jeju:
	pandoc -s --bibliography=biblio.bib --biblatex jeju.md -t beamer -o jeju.tex
	# open jeju.tex
	pdflatex jeju.tex
	# biber jeju
	# pdflatex jeju
	open jeju.pdf

hawaii:
	pandoc -s --bibliography=biblio.bib --biblatex hawaii.md -t beamer -o hawaii.tex
	# open hawaii.tex
	pdflatex hawaii.tex
	# biber hawaii
	# pdflatex hawaii
	open hawaii.pdf

aaai:
	pandoc -s --bibliography=biblio.bib --biblatex aaai2019-ktm-slides.md -t beamer -o aaai2019-ktm-slides.tex
	# open aaai2019-ktm-slides.tex
	pdflatex aaai2019-ktm-slides.tex
	# biber aaai2019-ktm-slides
	# pdflatex aaai2019-ktm-slides
	open aaai2019-ktm-slides.pdf

simple:
	pandoc -s aaai.md -t beamer -o aaai.pdf
	# open aaai.tex
	# pdflatex aaai
	open aaai.pdf

kyodai:
	pandoc -s --bibliography=biblio.bib --biblatex kyodai.md -t beamer -o kyodai.tex
	pdflatex kyodai
	# biber kyodai
	# pdflatex kyodai
	open kyodai.pdf

clean:
	rm aaai2019-ktm-slides.aux
