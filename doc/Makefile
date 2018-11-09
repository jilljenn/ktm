all:
	pandoc -s --bibliography=biblio.bib --biblatex kyodai.md -t beamer -o kyodai.tex
	pdflatex kyodai
	# biber kyodai
	# pdflatex kyodai
	open kyodai.pdf

tuto:
	pandoc -s --bibliography=biblio.bib --biblatex tuto.md -t beamer -o tuto.tex
	pdflatex tuto
	biber tuto
	pdflatex tuto
	open tuto.pdf
