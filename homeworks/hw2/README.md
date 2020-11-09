# Homework 2: Flow Models

**Due February 25, 11:59pm on Gradescope**

In this homework, you will implement and train a variety of different flow models. The homework consists of two components: 
* `hw2_notebook.ipynb` : Contains all of the coding questions, and will automatically generate and display results for you after completing each question. 
You will submit the notebook to Gradescope after completing the homework. 
Open it on Colab by clicking on the file, and then "Open in Colab" at the top. 
**Submit a PDF version of the notebook to the code (Print -> Preview -> Save) on Gradescope in the assignment with (code)**
* `hw2_latex` :  Contains LaTeX files and figures needed to generate your PDF submission to Gradescope. Copy the images saved from the notebook into the `figures/` folder.
**Submit the Latex PDF in the assignment with (PDF)**

You can open the notebook in Google Colab to get access to a free GPU, or you can link Colab to a local runtime to run it on your own GPU.  

## Solutions

To run on home comp:
```python
python -m homeworks.hw2.main [ex] --flags
```
where the flags are defined as
```python
parser.add_argument("ex", type=str, choices=["q1_a", "q1_b", "q2", "q3_a", "q3_b"],
                    help="Code of hw to do.")

parser.add_argument("--ds", type=int, choices=[1, 2],
                    help="Code of dataset to use.")

parser.add_argument("--gpu", action="store_true",
                    help="Training on gpu.")

parser.add_argument("--reload", action="store_true",
                    help="Reload model from previously saved state.")

parser.add_argument("--notrain", action="store_true",
                    help="Do not train model.")

parser.add_argument("--short", action="store_true",
                    help="Short training of one epoch only (otherwise use the default of each exercise).")

parser.add_argument("--screen", action="store_true",
                    help="Print stdout and stderr to screen (instead of log file in results folder).")

parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate for training.")

parser.add_argument("--epochs", type=int, default=15,
                    help="Max number of epochs.")

parser.add_argument("--bs", type=int, default=128,
                    help="Batch size.")
```

To run on **baobab** first log in `ssh gregorom@baobab2.hpc.unige.ch`. 
Once there edit `gjob` for gpu jobs or `cjob` for cpu jobs as you see fit and execute `sbatch ./gjob` or `sbatch ./cjob`.
Use `squeue -u $USER` to monitor the jobs.

The outputs are in `../hw2/results`. To get them to your comp do `rsync -azp gregorom@baobab2.hpc.unige.ch:/home/gregorom/hw2/results/ hw2/results
`
