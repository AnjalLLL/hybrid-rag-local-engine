"""Hand-verified R reference snippets for topics the local corpus doesn't cover well.

Hybrid retrieval can only surface what's in data/, and the corpus has no examples of
multinom/nnet, the cluster package, influence diagnostics beyond plot(model), or
ISLR2 usage. When a question hits one of those gaps, the model falls back to its own
"standard R knowledge" (see build_prompt's rules) and has been observed inventing
function arguments that don't exist (e.g. fviz_cluster(repel_points=...)). Each entry
below is deliberately narrow and only lists arguments that are real.
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
            "Multinomial logistic regression is NOT glm(..., family = \"multinomial\") "
            "(that family does not exist in base glm). Use nnet::multinom() instead. "
            "The response column must be a factor with 3+ levels."
        ),
    ),
    "kmeans_plot": ReferenceEntry(
        keywords=("kmeans", "k-means", "cluster center", "cluster centre"),
        required_library="# base R only, no package required",
        canonical_skeleton=(
            "km <- kmeans(df, centers = k)\n"
            "plot(df, col = km$cluster, pch = 19, main = \"K-means clustering\")\n"
            "points(km$centers, pch = 8, cex = 2, col = \"black\")  # adds cluster centers"
        ),
        pitfalls=(
            "Prefer base R plot()+points() since that is what the course material actually "
            "demonstrates. If you use factoextra::fviz_cluster(km, data = df) instead, the "
            "ONLY real arguments are: data, geom, ellipse.type, repel (TRUE/FALSE, not "
            "'repel_points'), main. There is no 'show_clusters' argument on fviz_cluster."
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
        pitfalls="pam/agnes/silhouette all come from the cluster package, not base R or factoextra.",
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
        pitfalls="Do not invent argument names for plot(model); its diagnostic panels are fixed by `which =`.",
    ),
    "normality_missing": ReferenceEntry(
        keywords=("shapiro", "shapiro.test", "normality", "normal distribution"),
        required_library="# base R (stats package, always loaded)",
        canonical_skeleton="shapiro.test(na.omit(df$x))",
        pitfalls=(
            "shapiro.test() errors on NA values. Always wrap the vector in na.omit() (or "
            "filter with complete.cases() first) before passing it in."
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
            "level/value, not the numeric summary."
        ),
    ),
    "islr2": ReferenceEntry(
        keywords=("islr2", "islr"),
        required_library="library(ISLR2)",
        canonical_skeleton="library(ISLR2)\ndata(Smarket)  # or whichever ISLR2 dataset the question names",
        pitfalls="ISLR2 datasets must be loaded with data(<name>) after library(ISLR2); do not read them from a CSV.",
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
            "Column names inside aes() are unquoted (aes(x = Sepal.Length), not "
            "aes(x = \"Sepal.Length\")). Real geoms include geom_point, geom_boxplot, "
            "geom_bar, geom_histogram, geom_line -- there is no geom_cluster or geom_kmeans."
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
            "confusionMatrix() requires both arguments to be factors with identical levels "
            "(same order) -- passing raw numeric predictions or probabilities directly errors."
        ),
    ),
    "decision_tree": ReferenceEntry(
        keywords=("rpart", "decision tree", "rpart.plot"),
        required_library="library(rpart)\nlibrary(rpart.plot)",
        canonical_skeleton=(
            "library(rpart)\n"
            "model <- rpart(y ~ ., data = df, method = \"class\")\n"
            "library(rpart.plot)\n"
            "rpart.plot(model)"
        ),
        pitfalls=(
            "method = \"class\" is required for a classification tree (use \"anova\" for "
            "regression trees). rpart.plot() is a separate package from rpart itself."
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
            "The response column must be a factor for classification (randomForest() runs "
            "regression if y is numeric). importance = TRUE must be set before calling "
            "importance()/varImpPlot()."
        ),
    ),
    "knn": ReferenceEntry(
        keywords=("knn", "k-nearest", "nearest neighbour", "nearest neighbor"),
        required_library="library(class)",
        canonical_skeleton="library(class)\npred <- knn(train = train_x, test = test_x, cl = train_y, k = k)",
        pitfalls=(
            "knn() takes train/test/cl/k in that order; predictors should be scaled first "
            "with scale(), and train_x/test_x must exclude the response column."
        ),
    ),
    "lda_qda": ReferenceEntry(
        keywords=("lda", "qda", "discriminant"),
        required_library="library(MASS)",
        canonical_skeleton="library(MASS)\nmodel <- lda(y ~ ., data = df)\npredict(model, newdata = df)$class",
        pitfalls="predict.lda() returns a list; the predicted labels are in $class, not the object itself.",
    ),
    "roc_curve": ReferenceEntry(
        keywords=("roc", "auc", "proc", "rocr"),
        required_library="library(pROC)  # or library(ROCR)",
        canonical_skeleton=(
            "library(pROC)\n"
            "roc_obj <- roc(response = test$y, predictor = pred_prob)\n"
            "plot(roc_obj)\n"
            "auc(roc_obj)"
        ),
        pitfalls=(
            "pROC::roc() needs predicted probabilities, not class labels. Don't mix pROC and "
            "ROCR syntax in one script -- ROCR uses prediction()/performance() instead."
        ),
    ),
    "pca_facto": ReferenceEntry(
        keywords=("factominer", "pca", "principal component"),
        required_library="library(FactoMineR)\nlibrary(factoextra)",
        canonical_skeleton=(
            "library(FactoMineR)\n"
            "pca_res <- PCA(df, graph = FALSE)\n"
            "library(factoextra)\n"
            "fviz_pca_ind(pca_res)\n"
            "fviz_eig(pca_res)"
        ),
        pitfalls=(
            "Base R alternative is prcomp(df, scale. = TRUE). PCA plots use "
            "fviz_pca_ind/fviz_pca_var/fviz_eig -- fviz_cluster is for clustering, not PCA."
        ),
    ),
    "igraph_network": ReferenceEntry(
        keywords=("igraph", "network", "social network", "graph.adjacency"),
        required_library="library(igraph)",
        canonical_skeleton="library(igraph)\ng <- graph_from_data_frame(edges_df, directed = FALSE)\nplot(g)\ndegree(g)",
        pitfalls=(
            "graph_from_data_frame() is the current function name (graph.data.frame() is a "
            "deprecated alias). plot() on an igraph object uses igraph's own args "
            "(vertex.size, vertex.label), not ggplot2 aesthetics."
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
            "apriori() needs a transactions object, not a data frame -- convert first with "
            "as(df, \"transactions\"). Thresholds go inside parameter = list(supp = ..., "
            "conf = ...), not as top-level arguments."
        ),
    ),
    "corr_plot": ReferenceEntry(
        keywords=("corrplot", "correlation matrix", "correlation plot"),
        required_library="library(corrplot)",
        canonical_skeleton=(
            "corr_matrix <- cor(df[, sapply(df, is.numeric)])\n"
            "library(corrplot)\n"
            "corrplot(corr_matrix, method = \"circle\")"
        ),
        pitfalls="cor() only accepts numeric columns; subset with sapply(df, is.numeric) first or it errors on factors/characters.",
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
            "Valid svm() kernel values are only \"linear\", \"polynomial\", \"radial\", "
            "\"sigmoid\" -- don't invent other kernel names. For naiveBayes(), use "
            "e1071::naiveBayes(y ~ ., data = df); the separate naivebayes package has a "
            "different function, naive_bayes(), with a different API."
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
