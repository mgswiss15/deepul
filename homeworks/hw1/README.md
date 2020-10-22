# Homework 1: Autoregressive Models

**Due February 11, 11:59pm on Gradescope**

In this homework, you will implement and train a variety of different autoregressive models, such as MADE and PixelCNN. The homework consists of two components:
* `hw1_notebook.ipynb` : Contains all of the coding questions, and will automatically generate and display results for you after completing each question. You will submit the notebook to Gradescope after completing the homework. Open it on Colab by clicking on the file, and then "Open in Colab" at the top. **Submit a PDF version of the notebook to the code (Print -> Preview -> Save) on Gradescope in the assignment with (code)**
* `hw1_latex` :  Contains LaTeX files and figures needed to generate your PDF submission to Gradescope. Copy the images saved from the notebook into the `figures/` folder and fill out the empty test losses.  **Submit the Latex PDF in the assignment with (PDF)**

You can open the notebook in Google Colab to get access to a free GPU, or you can link Colab to a local runtime to run it on your own GPU.  


## Solutions

>> MG: 12/10/2020: Dropped devs in jupyter notebook.

To run on home comp:
```python
python -m homeworks.hw1.main [ex] [ds] --gpu
```
where the flags are defined as
```python
parser.add_argument("--gpu", action="store_true",
                    help="Training on gpu.")

parser.add_argument("ex", type=str, choices=["q1a", "q1b", "q2a", "q2b", "q3a", "q3b", "q3c", "g3d"],
                    help="Code of hw to do.")

parser.add_argument("ds", type=int, choices=[1, 2],
                    help="Code of dataset to use.")
```

To run on **baobab** first log in `ssh gregorom@baobab2.hpc.unige.ch`. 
Once there edit `gjob` for gpu jobs or `cjob` for cpu jobs as you see fit and execute `sbatch ./gjob` or `sbatch ./cjob`.
Use `squeue -u $USER` to monitor the jobs.
