all: report.tex refer.bib iitk-logo.png
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

.Phony: clean
clean:
	rm -f report.aux report.bbl report.log report.blg;
