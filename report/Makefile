all: final
ORIG = report.tex refer.bib
IMG = iitk-logo.png
GEN = report.aux report.bbl report.log report.blg

$(GEN): $(ORIG) $(IMG)
	pdflatex report.tex
	bibtex report
	pdflatex report.tex

final: $(GEN) $(ORIG) $(IMG)
	pdflatex report.tex

.Phony: clean
clean:
	rm -f $(GEN);
