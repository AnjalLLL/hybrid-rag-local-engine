"""Hand-verified R reference snippets, checked against the actual ingested corpus.

Every entry below was checked with a direct grep over faiss_index/chunks.pkl (the full
extracted text of every PDF/PPTX/R file in data/, not just the .R scripts) for the literal
function/library call before being written or edited. Where a pattern is confirmed present,
the pitfalls text says which file/session it came from. Where something is NOT demonstrated
anywhere in the corpus, the pitfalls text says so explicitly instead of presenting it as the
taught approach -- retrieval can only surface what's actually in data/, and this file should
not silently substitute the author's own general R knowledge for that when a genuinely
different pattern is taught. Verified-taught patterns are preferred over "any correct R" ones
even when both are technically valid, since exam grading likely follows what was taught.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence


class ReferenceEntry(NamedTuple):
    keywords: Sequence[str]
    required_library: str
    canonical_skeleton: str
    pitfalls: str


REFERENCE_LIBRARY: Dict[str, ReferenceEntry] = {
    "multinom": ReferenceEntry(
        keywords=("multinom", "multinomial", "nnet"),
        required_library="library(nnet)",
        canonical_skeleton=(
            "library(nnet)\n"
            "model <- multinom(y ~ ., data = df)\n"
            "summary(model)\n"
            "pred <- predict(model, newdata = df)"
        ),
        pitfalls=(
            "Your materials use the nnet package, but only for fitting a feed-forward neural "
            "network via nnet(y ~ ., data = df, size = 5, maxit = 200, decay = 0.01) "
            "(R_Exam_Solutions_Both_Papers.pdf) -- multinom() itself is not demonstrated "
            "anywhere in your retrieved materials. It is still the standard, correct nnet "
            "function for multinomial logistic regression, so use it if the question is "
            "specifically about multinomial logistic regression, but treat this as general R "
            "knowledge rather than something shown verbatim in your notes. It is NOT "
            "glm(..., family = \"multinomial\") -- that family does not exist in base glm. "
            "The response column must be a factor with 3+ levels."
        ),
    ),
    "kmeans_plot": ReferenceEntry(
        keywords=("kmeans", "k-means", "cluster center", "cluster centre"),
        required_library="library(cluster)  # this is Session 30's own k-means visualization",
        canonical_skeleton=(
            "us <- scale(df)  # standardize: kmeans is distance-based, so unscaled variables "
            "with a larger numeric range would dominate the clustering\n"
            "set.seed(11)\n\n"
            "# Part with just the plot (verified from Session 30, \"Visualizing clusters\"):\n"
            "km <- kmeans(us, centers = k, nstart = 20)\n"
            "library(cluster)\n"
            "clusplot(us[, c(\"var1\", \"var2\")], km$cluster,\n"
            "         lines = 0, shade = TRUE, color = TRUE, labels = 2,\n"
            "         plotchar = FALSE, span = TRUE,\n"
            "         main = \"Cluster plot\", xlab = \"var1\", ylab = \"var2\")\n\n"
            "# A LATER part that adds cluster centers must redraw the SAME clusplot() call "
            "again before points() -- do not call points() by itself:\n"
            "clusplot(us[, c(\"var1\", \"var2\")], km$cluster,\n"
            "         lines = 0, shade = TRUE, color = TRUE, labels = 2,\n"
            "         plotchar = FALSE, span = TRUE,\n"
            "         main = \"Cluster plot with centers\", xlab = \"var1\", ylab = \"var2\")\n"
            "points(km$centers[, c(\"var1\", \"var2\")], pch = 8, cex = 2, col = \"black\")"
        ),
        pitfalls=(
            "Each lettered sub-part is knitted as its own separate code chunk, and knitr "
            "opens a fresh graphics device per chunk -- so a later sub-part that only calls "
            "points()/lines()/abline()/text() by itself, without first re-calling the plot "
            "function (clusplot()/plot()) in that SAME sub-part, fails with 'plot.new has not "
            "been called yet' (or silently renders nothing) when knitted, even though it may "
            "look fine if the whole script happens to run top-to-bottom in one go. ALWAYS "
            "redraw the base plot in the same code block right before adding anything on top "
            "of it, in every sub-part that needs it, even if an earlier sub-part already drew "
            "a similar plot. The clusplot() call itself, including every argument (lines, "
            "shade, color, labels, plotchar, span), is copied from Session 30's own k-means "
            "visualization on the iris dataset -- this is the actually-taught pattern, not a "
            "guess. A simpler base-R alternative also appears in your materials "
            "(exam_ref_unit5_unsupervised.R, PATTERN 6): plot(x, y, col = km$cluster, "
            "pch = 19) + points(km$centers) -- the same redraw-before-points() rule applies "
            "there too. factoextra::fviz_cluster() is NEVER demonstrated anywhere in your "
            "retrieved materials (checked directly -- zero occurrences, even though one "
            "unrelated file imports factoextra without ever calling an fviz_* function) -- do "
            "not present it as the taught approach; only use it if the question explicitly "
            "names factoextra or fviz_cluster. Either way, pick exactly TWO of the requested "
            "variables for the axes -- passing the whole multi-column data frame to plot() "
            "produces a scatterplot matrix, not a single graph, and clusplot() on 3+ raw "
            "columns switches to an internal PCA projection instead of the two named "
            "variables. scale() the data first when variables are on different measurement "
            "scales, and write an interpretation that names the concrete pattern, e.g. "
            "'cluster 1 = higher-crime states (higher Murder/Assault), cluster 2 = "
            "lower-crime states' -- not 'the data is divided into clusters'."
        ),
    ),
    "cluster_pkg": ReferenceEntry(
        keywords=("pam", "agnes", "silhouette", "medoids"),
        required_library="library(cluster)",
        canonical_skeleton=(
            "library(cluster)\n"
            "pam_res <- pam(df, k = k)\n"
            "sil <- silhouette(pam_res$clustering, dist(df))\n"
            "plot(sil)"
        ),
        pitfalls=(
            "library(cluster) does appear in your materials (Session 30), but only for "
            "clusplot() when visualizing k-means results -- pam()/agnes()/silhouette() "
            "themselves are not demonstrated anywhere in your retrieved materials. They are "
            "still standard, correct functions from the same package, so use them if the "
            "question specifically asks for pam/agnes/silhouette, but treat this as general R "
            "knowledge rather than something shown in your notes."
        ),
    ),
    "regression_diagnostics": ReferenceEntry(
        keywords=("influence", "cooks.distance", "cook's distance", "hatvalues", "leverage", "residual"),
        required_library="# base R (stats package, always loaded)",
        canonical_skeleton=(
            "model <- lm(y ~ ., data = df)\n"
            "par(mfrow = c(2, 2)); plot(model)  # residuals vs fitted, Q-Q, scale-location, leverage\n"
            "cooks.distance(model)\n"
            "hatvalues(model)\n"
            "influence.measures(model)"
        ),
        pitfalls=(
            "plot(model) itself is demonstrated in your materials "
            "(exam_ref_data_and_regression_basics.R) -- do not invent argument names for it, "
            "its diagnostic panels are fixed by `which =`. cooks.distance()/hatvalues()/"
            "influence.measures() are not demonstrated in your retrieved materials beyond "
            "that single plot(model) call; they are standard base-R stats functions, useful "
            "if the question asks for them specifically."
        ),
    ),
    "normality_missing": ReferenceEntry(
        keywords=("shapiro", "shapiro.test", "normality", "normal distribution"),
        required_library="# base R (stats package, always loaded)",
        canonical_skeleton="shapiro.test(na.omit(df$x))",
        pitfalls=(
            "shapiro.test() is demonstrated in your materials "
            "(exam_ref_statistical_tests.R and exam_ref_unit4_advanced_supervised.R). It "
            "errors on NA values, so always wrap the vector in na.omit() (or filter with "
            "complete.cases() first) before passing it in -- your materials use na.omit() "
            "this same way elsewhere before other tests/models."
        ),
    ),
    "type_functions": ReferenceEntry(
        keywords=("levels", "summary", "factor", "as.factor"),
        required_library="# base R",
        canonical_skeleton="df$x <- as.factor(df$x)\nlevels(df$x)\nsummary(df$x)",
        pitfalls=(
            "levels() only works on a factor, not a numeric or character vector -- convert "
            "with as.factor() first if needed. summary() on a numeric vector gives "
            "min/1Q/median/mean/3Q/max; on a factor or character vector it gives counts per "
            "level/value, not the numeric summary. These are base-R behaviors, not specific "
            "to any one file in your materials."
        ),
    ),
    "grouped_test_extraction": ReferenceEntry(
        keywords=("each category", "by month", "each group", "grouped test", "by category"),
        required_library="# base R (stats package, always loaded)",
        canonical_skeleton=(
            "shapiro_by_group <- by(df$x, df$group, shapiro.test)\n"
            "print(shapiro_by_group)  # inspect each group's W statistic and p-value directly\n"
            "shapiro_pvalues <- sapply(shapiro_by_group, function(res) res$p.value)\n"
            "print(shapiro_pvalues)\n"
            "all(shapiro_pvalues > 0.05)  # TRUE only if every group individually passed"
        ),
        pitfalls=(
            "by() returns an object of class 'by' (a list holding one test result per group) "
            "-- it has NO $p.value field of its own, so by(...)$p.value silently returns NULL "
            "instead of erroring. Using that directly in a condition, e.g. "
            "all(by(x, g, shapiro.test)$p.value > 0.05), is a real bug that will NOT show up "
            "as an execution error: all(NULL > 0.05) evaluates to TRUE regardless of the "
            "actual test outcomes, so any if/else branch guarded by it always takes the same "
            "path no matter what the per-group tests actually found. Always extract each "
            "group's p-value first with sapply(by_result, function(res) res$p.value) (or "
            "vapply/purrr::map_dbl) before using it in any comparison, condition, or all()/"
            "any() check."
        ),
    ),
    "anova_pvalue_extract": ReferenceEntry(
        keywords=("aov", "anova", "one-way anova", "tukeyhsd"),
        required_library="# base R (stats package, always loaded)",
        canonical_skeleton=(
            "anova_model <- aov(y ~ group, data = df)\n"
            "print(summary(anova_model))  # simplest: just read the p-value from the printed table\n\n"
            "# only if you need the p-value in code, e.g. for an if() branch:\n"
            "anova_pvalue <- summary(anova_model)[[1]][[\"Pr(>F)\"]][1]\n"
            "if (anova_pvalue < 0.05) TukeyHSD(anova_model)"
        ),
        pitfalls=(
            "summary(an_aov_model) returns a list containing one data frame, not a simple "
            "named object -- summary(model)$group or summary(model)$`group`$`Pr(>F)` are BOTH "
            "invalid. R has no method that lets you chain $Pr(>F) like a function call after "
            "$group; that syntax will error. The correct way to pull the p-value out "
            "programmatically is summary(anova_model)[[1]][[\"Pr(>F)\"]][1]. If you don't "
            "actually need to branch on it in code, it is simpler and safer to just "
            "print(summary(anova_model)) and read the p-value directly from the printed "
            "table instead of extracting it."
        ),
    ),
    "islr2": ReferenceEntry(
        keywords=("islr2", "islr"),
        required_library="library(ISLR2)",
        canonical_skeleton="library(ISLR2)\ndata(Smarket)  # or whichever ISLR2 dataset the question names",
        pitfalls=(
            "library(ISLR2) is used throughout ISLRv2_corrected_June_2023.pdf, your primary "
            "textbook (15+ occurrences). ISLR2 datasets must be loaded with data(<name>) "
            "after library(ISLR2); do not read them from a CSV. Note the ISLR2 package itself "
            "is not installed in this environment, so --verify-r execution will fail on "
            "library(ISLR2) with a missing-package error -- that's expected, not a code bug."
        ),
    ),
    "ggplot2": ReferenceEntry(
        keywords=("ggplot", "ggplot2", "geom_point", "geom_bar", "geom_boxplot", "aes"),
        required_library="library(ggplot2)",
        canonical_skeleton=(
            "ggplot(df, aes(x = var1, y = var2)) +\n"
            "  geom_point() +\n"
            "  labs(title = \"Title\", x = \"X\", y = \"Y\")"
        ),
        pitfalls=(
            "ggplot2 is heavily used across your materials (100+ occurrences, especially "
            "R-for-Data-Science.pdf and exam_ref_unit3_visualization.R). Column names inside "
            "aes() are unquoted (aes(x = Sepal.Length), not aes(x = \"Sepal.Length\")). Real "
            "geoms include geom_point, geom_boxplot, geom_bar, geom_histogram, geom_line -- "
            "there is no geom_cluster or geom_kmeans. Note that for k-means cluster plots "
            "specifically, your materials use base R/clusplot() instead of ggplot2 -- see the "
            "k-means reference entry."
        ),
    ),
    "caret_confusion": ReferenceEntry(
        keywords=("confusionmatrix", "caret", "traincontrol"),
        required_library="library(caret)",
        canonical_skeleton=(
            "library(caret)\n"
            "pred <- predict(model, newdata = test)\n"
            "confusionMatrix(as.factor(pred), as.factor(test$y))"
        ),
        pitfalls=(
            "confusionMatrix() is demonstrated in your materials (Madhu Ajit Pandey R "
            "Presentation, R_Exam_Solutions_Both_Papers.pdf, Session 23). It requires both "
            "arguments to be factors with identical levels (same order) -- passing raw "
            "numeric predictions or probabilities directly errors."
        ),
    ),
    "decision_tree": ReferenceEntry(
        keywords=("rpart", "decision tree", "rpart.plot"),
        required_library="library(rpart)",
        canonical_skeleton=(
            "library(rpart)\n"
            "model <- rpart(y ~ ., data = df, method = \"class\")\n"
            "pred <- predict(model, newdata = test, type = \"class\")\n"
            "printcp(model)\n\n"
            "# Plotting: your materials only demonstrate base R plot()+text() for this\n"
            "# (ISLR2 ch.8 tree.carseats/tree.boston examples, Session 28.1):\n"
            "plot(model)\n"
            "text(model, pretty = 0)"
        ),
        pitfalls=(
            "rpart.plot (the package) is never demonstrated anywhere in your retrieved "
            "materials -- do not present it as the taught approach. Your materials plot trees "
            "with base R's plot()+text(pretty=0) instead (verified in ISLR2's tree.carseats/"
            "tree.boston examples and Session 28.1, though those specific pages use the "
            "`tree` package's tree() function rather than rpart() -- the same plot()/text() "
            "S3 methods work on an rpart object too). rpart() itself with method = \"class\" "
            "is demonstrated in R_Exam_Solutions_Both_Papers.pdf; method = \"class\" is "
            "required for a classification tree (use \"anova\" for regression trees)."
        ),
    ),
    "random_forest": ReferenceEntry(
        keywords=("randomforest", "random forest"),
        required_library="library(randomForest)",
        canonical_skeleton=(
            "library(randomForest)\n"
            "model <- randomForest(y ~ ., data = df, ntree = 500, importance = TRUE)\n"
            "importance(model)\n"
            "varImpPlot(model)"
        ),
        pitfalls=(
            "randomForest()/varImpPlot() are demonstrated in your materials (ISLR2 ch.8, "
            "R_Exam_Solutions_Both_Papers.pdf). The response column must be a factor for "
            "classification (randomForest() runs regression if y is numeric). "
            "importance = TRUE must be set before calling importance()/varImpPlot()."
        ),
    ),
    "knn": ReferenceEntry(
        keywords=("knn", "k-nearest", "nearest neighbour", "nearest neighbor"),
        required_library="library(class)  # classification only -- see below for regression",
        canonical_skeleton=(
            "# KNN CLASSIFICATION (categorical/factor response), verified in your materials\n"
            "# (ISLR2 ch.4):\n"
            "library(class)\n"
            "pred <- knn(train = train_x, test = test_x, cl = train_y, k = k)\n\n"
            "# KNN REGRESSION (continuous/numeric response), verified in your materials\n"
            "# (exam_ref_supervised_regression.R, PATTERN 2):\n"
            "library(caret)\n"
            "knn_model <- knnreg(y ~ ., data = train, k = k)\n"
            "pred <- predict(knn_model, newdata = test)"
        ),
        pitfalls=(
            "These are two different functions for two different tasks -- check which one "
            "the question actually asks for before picking. class::knn() is CLASSIFICATION "
            "ONLY: cl must be a factor of class labels, and it takes train/test/cl/k in that "
            "order (predictors scaled first with scale(), train_x/test_x excluding the "
            "response column). If the question says 'KNN regression' or the response is a "
            "continuous/numeric variable (e.g. predicting mpg, price, temperature), "
            "class::knn() is the WRONG function -- use caret::knnreg(formula, data, k) "
            "instead, exactly as demonstrated in exam_ref_supervised_regression.R."
        ),
    ),
    "lda_qda": ReferenceEntry(
        keywords=("lda", "qda", "discriminant"),
        required_library="library(MASS)",
        canonical_skeleton="library(MASS)\nmodel <- lda(y ~ ., data = df)\npredict(model, newdata = df)$class",
        pitfalls=(
            "lda()/qda() are demonstrated in your materials (ISLR2 ch.4, Classification_Models "
            "presentation). predict.lda() returns a list; the predicted labels are in $class, "
            "not the object itself."
        ),
    ),
    "roc_curve": ReferenceEntry(
        keywords=("roc", "auc", "proc", "rocr"),
        required_library="library(pROC)  # or library(ROCR) -- both appear in your materials",
        canonical_skeleton=(
            "library(pROC)\n"
            "roc_obj <- roc(response = test$y, predictor = pred_prob)\n"
            "plot(roc_obj)\n"
            "auc(roc_obj)"
        ),
        pitfalls=(
            "Both APIs are demonstrated in your materials: pROC::roc() (Madhu Ajit Pandey "
            "presentation, Session 27) and ROCR::prediction()/performance() (ISLR2 ch.9, "
            "Session 25 and 27) -- either is fine, but pROC::roc() needs predicted "
            "probabilities, not class labels, and you must not mix pROC and ROCR syntax "
            "in one script."
        ),
    ),
    "pca_facto": ReferenceEntry(
        keywords=("factominer", "pca", "principal component"),
        required_library="library(FactoMineR)  # fit with this; visualize with base R, not factoextra",
        canonical_skeleton=(
            "library(FactoMineR)\n"
            "pca_res <- PCA(df, graph = FALSE)\n\n"
            "# Visualization: your materials use base R for this, not factoextra's fviz_* functions\n"
            "pca_base <- prcomp(df, scale. = TRUE)\n"
            "screeplot(pca_base, type = \"lines\", main = \"Scree Plot\")\n"
            "biplot(pca_base)"
        ),
        pitfalls=(
            "FactoMineR::PCA() is demonstrated in your materials (PCA_Anjal_RollNo3.R, Madhu "
            "Ajit Pandey presentation), but factoextra's fviz_pca_ind/fviz_pca_var/fviz_eig "
            "are NEVER used anywhere in your retrieved materials (checked directly -- zero "
            "occurrences, even in the one file that imports factoextra). Your materials "
            "instead visualize PCA with base R screeplot(pca_model, type = \"lines\") and "
            "biplot(pca_model) (exam_ref_unit5_unsupervised.R, exam_ref_pca_clustering.R, and "
            "the ISLR2 textbook) -- use those instead of fviz_* calls unless the question "
            "explicitly names factoextra."
        ),
    ),
    "igraph_network": ReferenceEntry(
        keywords=("igraph", "network", "social network", "graph.adjacency"),
        required_library="library(igraph)",
        canonical_skeleton="library(igraph)\ng <- graph.data.frame(edges_df, directed = FALSE)\nplot(g)\ndegree(g)",
        pitfalls=(
            "graph.data.frame() is what Session 18 actually uses (2 occurrences) -- match "
            "that spelling even though it's technically a deprecated alias of the newer "
            "graph_from_data_frame(), which also appears once in "
            "exam_ref_unit3_visualization.R if you prefer the modern name; both work "
            "identically. plot() on an igraph object uses igraph's own args (vertex.size, "
            "vertex.label), not ggplot2 aesthetics."
        ),
    ),
    "arules_assoc": ReferenceEntry(
        keywords=("apriori", "arules", "association rule"),
        required_library="library(arules)",
        canonical_skeleton=(
            "library(arules)\n"
            "txns <- as(df, \"transactions\")\n"
            "rules <- apriori(txns, parameter = list(supp = 0.1, conf = 0.8))\n"
            "inspect(sort(rules, by = \"lift\")[1:5])"
        ),
        pitfalls=(
            "apriori() is demonstrated in your materials (Session 31, "
            "exam_ref_unit5_unsupervised.R). It needs a transactions object, not a data frame "
            "-- convert first with as(df, \"transactions\") (or build a list of item vectors "
            "directly, as exam_ref_unit5_unsupervised.R does). Thresholds go inside "
            "parameter = list(supp = ..., conf = ...), not as top-level arguments."
        ),
    ),
    "corr_plot": ReferenceEntry(
        keywords=("corrplot", "correlation matrix", "correlation plot"),
        required_library="# base R -- the corrplot package is never used in your materials",
        canonical_skeleton=(
            "corr_matrix <- cor(df[, sapply(df, is.numeric)])\n"
            "print(round(corr_matrix, 2))\n"
            "pairs(df[, sapply(df, is.numeric)])  # base R scatterplot matrix, seen in ISLR2"
        ),
        pitfalls=(
            "The corrplot package is NEVER demonstrated anywhere in your retrieved materials "
            "(checked directly -- zero occurrences) -- do not present corrplot() as the "
            "taught approach for visualizing correlations. Your materials (ISLR2) use base R "
            "pairs() for a scatterplot matrix instead, alongside a plain cor() table. cor() "
            "only accepts numeric columns; subset with sapply(df, is.numeric) first or it "
            "errors on factors/characters."
        ),
    ),
    "svm_bayes": ReferenceEntry(
        keywords=("svm", "support vector", "naivebayes", "naive bayes"),
        required_library="library(e1071)",
        canonical_skeleton=(
            "library(e1071)\n"
            "model <- svm(y ~ ., data = df, kernel = \"radial\", probability = TRUE)\n"
            "pred <- predict(model, newdata = df)"
        ),
        pitfalls=(
            "svm()/naiveBayes() are both demonstrated in your materials (ISLR2 ch.9, "
            "R_Exam_Solutions_Both_Papers.pdf). Valid svm() kernel values are only "
            "\"linear\", \"polynomial\", \"radial\", \"sigmoid\" -- don't invent other kernel "
            "names. For naiveBayes(), use e1071::naiveBayes(y ~ ., data = df); the separate "
            "naivebayes package (with a different function, naive_bayes()) is not used "
            "anywhere in your materials -- stick with e1071::naiveBayes()."
        ),
    ),
}


def match_reference_entries(topic_tokens: Sequence[str]) -> List[ReferenceEntry]:
    """Return reference entries whose keywords appear among the extracted topic tokens."""

    token_set = {token.lower() for token in topic_tokens}
    matches: List[ReferenceEntry] = []
    for entry in REFERENCE_LIBRARY.values():
        if any(keyword in token_set or _keyword_in_tokens(keyword, token_set) for keyword in entry.keywords):
            matches.append(entry)
    return matches


def _keyword_in_tokens(keyword: str, token_set: set) -> bool:
    """Match multi-word keywords (e.g. 'k-means') against single-word BM25-style tokens."""

    parts = keyword.replace("-", " ").replace(".", " ").split()
    if len(parts) == 1:
        return parts[0] in token_set
    return all(part in token_set for part in parts)


def format_reference_block(entries: Sequence[ReferenceEntry]) -> str:
    """Render matched reference entries as a prompt-ready block."""

    if not entries:
        return ""

    sections = []
    for entry in entries:
        sections.append(
            f"{entry.required_library}\n{entry.canonical_skeleton}\nNote: {entry.pitfalls}"
        )
    return "\n\n".join(sections)
