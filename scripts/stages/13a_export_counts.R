# Export the deposited GSE211785 Visium count matrix to Matrix Market for Python.
suppressMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
tmp <- tempfile(fileext = ".rds")
system2("bash", c("-c", shQuote(sprintf("zcat %s > %s", src, tmp))))
x <- readRDS(tmp)
if (!inherits(x, "dgCMatrix")) {
  if (isS4(x) && "assays" %in% slotNames(x)) x <- x@assays$Spatial@counts else x <- as(x, "dgCMatrix")
}
writeMM(x, file.path(out, "counts.mtx"))
writeLines(rownames(x), file.path(out, "genes.tsv"))
writeLines(colnames(x), file.path(out, "barcodes.tsv"))
cat(dim(x), "\n")
