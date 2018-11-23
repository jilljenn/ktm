all:
	pandoc -s --bibliography=biblio.bib --biblatex aaai2019-ktm-slides.md -t beamer -o aaai2019-ktm-slides.tex
	pdflatex aaai2019-ktm-slides
	# biber aaai2019-ktm-slides
	# pdflatex aaai2019-ktm-slides
	open aaai2019-ktm-slides.pdf


kyodai:
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

clean:
	rm aaai2019-ktm-slides.aux
