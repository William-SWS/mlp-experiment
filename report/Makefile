relatorio.pdf: relatorio.tex $(wildcard sections/*.tex) $(wildcard tables/*.tex)
	latexmk -pdf -interaction=nonstopmode -halt-on-error relatorio.tex

clean:
	latexmk -c relatorio.tex

.PHONY: clean
