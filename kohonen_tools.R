# This script provides wrappers around the toolbox provided by kohonen package
# Including non-standard distance measures, diagnostic plotting in grid format
# and cross-validation tools

  require(tidyverse)
  require(kohonen)
  require(Rcpp)
  require(grDevices)
  require(rlang)
  require(cowplot)
  require(caret)
  require(furrr)
  
# helper functions -----
  
  set_colors_ <- function(color_no, seed = 123) {
    
    ## picks n colors at random from the standard palette
    
    set.seed(seed)
    
    return(colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)] %>% 
             sample(size = color_no))
    
  }
  
  create_folds_ <- function(data, n, seed = 123) {
    
    ## creates n data folds
    
    set.seed(seed)
    
    fold_ids <- createFolds(y = rownames(data), 
                            k = n, 
                            list = T, 
                            returnTrain = F)
    
    return(fold_ids)
    
  }
  
  validate_fold <- function(kohonen_obj, obs_ids, seed = 123) {
    
    ## compares the real unit assignment for the obs_ids
    ## with the one predicted with the rest data
    
    true_outcome <- kohonen_obj$unit.classif[obs_ids] ## true outcome: fitted with the kohonen object
    
    ## fittting a new kohonen object to the remaining data
    
    train_data <- kohonen_obj$data[[1]][-obs_ids, ]
    
    test_data <- kohonen_obj$data[[1]][obs_ids, , drop = F]
    
    new_som <- fit_som(data = train_data, 
                       grid = kohonen_obj$grid, 
                       rlen = nrow(kohonen_obj$changes), 
                       alpha = kohonen_obj$alpha, 
                       radius = kohonen_obj$radius, 
                       user.weights = kohonen_obj$user.weights, 
                       whatmap = kohonen_obj$whatmap, 
                       maxNA.fraction = kohonen_obj$maxNA.fraction, 
                       dist.fcts = kohonen_obj$dist.fcts, 
                       init = kohonen_obj$codes)
    
    ## making predictions
    
    predictions <- kohonen::map(x = new_som, 
                                newdata = test_data)
    
    return(tibble(y = true_outcome, 
                  .fitted = predictions$unit.classif))
    
  }
  
# C++ distance kernels ------
  
  kernels <- list()
  
  kernels$BCcode <- '#include <Rcpp.h>
    typedef double (*DistanceFunctionPtr)(double *, double *, int, int);

    double brayCurtisDissim(double *data, double *codes, int n, int nNA) {
      if (nNA > 0) return NA_REAL;

      double num = 0.0, denom = 0.0;
      
      for (int i = 0; i < n; i++) {
        num += std::abs(data[i] - codes[i]);
        denom += data[i] + codes[i];
      }

      return num/denom;
        
    }

  // [[Rcpp::export]]
  Rcpp::XPtr<DistanceFunctionPtr> BrayCurtis() {
    return Rcpp::XPtr<DistanceFunctionPtr>(
    new DistanceFunctionPtr(&brayCurtisDissim));
  }'
  
  kernels$smc_code <- '#include <Rcpp.h>
    typedef double (*DistanceFunctionPtr)(double *, double *, int, int);

    double mySMCdef(double *data, double *codes, int n, int nNA) {
      if (nNA > 0) return NA_REAL;

      double num = 0.0;
      
      for (int i = 0; i < n; i++) {
        num += std::abs(data[i] - codes[i]);
      }

      return num/n;
        
    }

  // [[Rcpp::export]]
  Rcpp::XPtr<DistanceFunctionPtr> SMC() {
    return Rcpp::XPtr<DistanceFunctionPtr>(
    new DistanceFunctionPtr(&mySMCdef));
  }'
  
  kernels$jaccard_code <- '#include <Rcpp.h>
    typedef double (*DistanceFunctionPtr)(double *, double *, int, int);

    double myJaccardDef(double *data, double *codes, int n, int nNA) {
      if (nNA > 0) return NA_REAL;

      double num = 0.0;
      double denom1 = 0.0;
      double denom2 = 0.0;
      double jacc = 0.0;
      
      for (int i = 0; i < n; i++) {
        num += data[i] * codes[i];
        denom1 += data[i] * data[i];
        denom2 += codes[i] * codes[i];
      }
      
      jacc = 1 - num/(denom1 + denom2 - num);

      return jacc;
        
    }

  // [[Rcpp::export]]
  Rcpp::XPtr<DistanceFunctionPtr> Jaccard() {
    return Rcpp::XPtr<DistanceFunctionPtr>(
    new DistanceFunctionPtr(&myJaccardDef));
  }'
  
  kernels %>% 
    walk(function(x) sourceCpp(code = x))

  
  
  
# SOM fitting and diagnostic plots -----
  
  fit_som <- function(data, seed = 123, ...) {
    
    ## sets the seed and fits a Kohonen object
    
    set.seed(seed)
    
    kohonen_obj <- som(as.matrix(data), ...)
    
    return(kohonen_obj)
    
  }
  
  plot_som <- function(kohonen_object) {
    
    ## generates diagnostic plots for the test cohort
    
    if(class(kohonen_object) != 'kohonen') {
      
      stop('The input has to be a kohonen object')
      
    }
    
    plot_exprs <- enexpr(kohonen_object)
   
    plot_exprs_lst <- c('change', 
                        'codes', 
                        'counts', 
                        'mapping', 
                        'dist.neighbours') %>% 
      purrr::map(function(x) expr(~plot(!!plot_exprs, !!x))) %>% 
      set_names(c('change', 
                  'codes', 
                  'counts', 
                  'mapping', 
                  'dist.neighbours'))
    
    #return(plot_exprs_lst)

    diagn_plots <- plot_exprs_lst %>% 
      purrr::map(function(x) ggdraw(as_grob(eval_tidy(x)))) #%>% 
     # purrr::map(ggdraw)

    return(diagn_plots)
    
  }
  
# Cross-validation of the Kohonen objects -----
  
  cv_som <- function(kohonen_obj, nfolds = 10, seed = 123) {
    
    ## performs cross-validation of the kohonen object
    
    id_lst <- create_folds_(data = kohonen_obj$data[[1]], 
                            n = nfolds, 
                            seed = seed)
    
    cv_results <- id_lst %>% 
      purrr::map(validate_fold, 
                 kohonen_obj = kohonen_obj, 
                 seed = seed) %>% 
      purrr::map2_dfr(., 
                      names(.), 
                      function(x, name_x) mutate(x, fold = name_x)) %>% 
      mutate(correct = as.numeric(y == .fitted))
    
    ## correct and false rate
    
    pred_stats <- tibble(nfolds = nfolds, 
                         corr_rate = sum(cv_results$correct)/nrow(cv_results), 
                         error_rate = 1 - sum(cv_results$correct)/nrow(cv_results))
    
    return(list(cv_results = cv_results, 
                cv_stats = pred_stats))
    
  }
  
# Extraction functions -----
  
  get_node_ass <- function(kohonen_object) {
    
    ## extracts the node assignment information
    ## and the distance to the winning unit (neuron) vector
    
    if(class(kohonen_object) != 'kohonen') {
      
      stop('The input has to be a kohonen object')
      
    }
    
    return(tibble(feature = rownames(kohonen_object$data[[1]]), 
                  node = factor(kohonen_object$unit.classif), 
                  neuro_dist = kohonen_object$distances))
    
  }
  
# Advanced plots ------
  
  plot_radial_dist <- function(kohonen_object, 
                               highlight_features = NULL, 
                               feature_labs = NULL,
                               fill_colors = NULL, 
                               plot_title = NULL, 
                               plot_subtitle = NULL, 
                               plot_tag = NULL, 
                               cust_theme = NULL, 
                               seed = 123) {
    
    ## plots a series of radial plots, one for each unit,
    ## with the distances of the features from the winning unit (neuron) vector
    
    ## enables labeling of the selected features
    
    node_ass_tbl <- get_node_ass(kohonen_object = kohonen_object)
    
    if(all(!is.null(highlight_features) & !is.null(feature_labs))) {
      
      node_ass_tbl <- node_ass_tbl %>% 
        mutate(feat_lab = set_names(feature_labs, 
                                    highlight_features)[feature])
      
    } else {
      
      node_ass_tbl <- node_ass_tbl %>% 
        mutate(feat_lab = NA)
      
    }

    ## theme
    
    if(is.null(cust_theme)) {
      
      cust_theme <- theme_classic()
      
    }
    
    radial_plot <- node_ass_tbl %>% 
      ggplot(aes(x = neuro_dist, 
                 y = reorder(feature, neuro_dist), 
                 fill = node)) + 
      geom_segment(aes(x = 0, 
                       xend = neuro_dist, 
                       yend = reorder(feature, neuro_dist))) + 
      geom_point(size = 2, 
                 shape = 21) +
      geom_label_repel(aes(label = feat_lab), 
                       size = 2.4, 
                       label.padding = 0.1, 
                       box.padding = 0.1) + 
      facet_wrap( ~ node) + 
      coord_polar(theta = 'y') + 
      cust_theme + 
      theme(axis.text.x = element_blank(), 
            panel.grid.major.x = element_line(color = 'gray60'), 
            axis.title = element_blank()) + 
      labs(title = plot_title, 
           subtitle = plot_subtitle, 
           tag = plot_tag)
    
    if(is.null(fill_colors)) {
      
      radial_plot <- radial_plot + 
        scale_fill_manual(values = set_colors_(color_no = kohonen_object$grid$xdim * kohonen_object$grid$ydim, 
                                               seed = seed)) + 
        guides(fill = F)
      
    } else {
      
      radial_plot <- radial_plot + 
        scale_fill_manual(values = fill_colors)
      
    }
    
    return(radial_plot)
    
  }
  

  
# END ---