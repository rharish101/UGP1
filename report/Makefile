all: final

ORIG = report.tex refer.bib
GEN = report.aux report.bbl report.log report.blg report.out

IMG_DIR := images
IMG_LIST := iitk-logo.png adv-loss.png cont-loss.png ovrl-loss.png psnr.png ssim.png overlap.png mad-srgan-gen-arch.png mad-srgan-disc-arch.png
IMGS := $(foreach img, $(IMG_LIST), $(IMG_DIR)/$(img))

$(GEN): $(ORIG) $(IMGS)
	pdflatex report.tex
	bibtex report
	pdflatex report.tex

final: $(GEN) $(ORIG) $(IMGS)
	pdflatex report.tex

.Phony: clean
clean:
	rm -f $(GEN)
