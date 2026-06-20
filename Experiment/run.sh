#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p html_output
mkdir -p plots

# List of files to process (in order) - mix of .Rmd and .ipynb
files=(
 # "code/01_H1_bayesian_priors.ipynb"
  #"code/02_H1_bayes_poisson_model.Rmd"
  "code/03_H1_GGLM_poisson_model.Rmd"
  #"code/04_H2_wilcoxon.Rmd"
  #"code/05_Exploratory_entropy.ipynb"
  #"code/06_Plots.ipynb"
)

# Run each file
for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo "Processing: $file"
    
    if [[ "$file" == *.Rmd ]]; then
      # Process R Markdown files
      Rscript -e "rmarkdown::render('$file', output_dir = 'html_output')"
      
    elif [[ "$file" == *.ipynb ]]; then
      # Process Jupyter notebooks
      jupyter nbconvert --to html --execute "$file" --output-dir html_output
      
    fi
    
    if [ $? -eq 0 ]; then
      echo "✓ Completed: $file"
    else
      echo "✗ Failed: $file"
      exit 1
    fi
  else
    echo "⚠ File not found: $file"
  fi
done

echo ""
echo "All files processed! HTML outputs saved to html_output/"