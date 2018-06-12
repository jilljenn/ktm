all:
	pandoc -s --bibliography=biblio.bib --biblatex tuto.md -t beamer -o tuto.tex
	pdflatex tuto
	biber tuto
	pdflatex tuto
	open tuto.pdf
