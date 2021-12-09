# AUD Final Project
This is the final project for MGMT 590 - Analyzing Unstructured Data. **Instructor:** Jinyang Zheng

## Premise
Craigslist is an online advertising host and one of the largest peer-to-peer advertising websites in the world. It serves
over 570 cities in more than 70 countries.

One of the shortcomings of Craigslist is its lack of filtering methods for different categories. Specifically, we wanted to 
target the Jobs/Resumes sections of the website. Currently, jobs are categorized into more than 25 different sections, while
resumes have no classification.

In order to increase the visibility for both employers and prospective employees, we wanted to use NLP to create a classification
model which would segment the resumes into the categories that already exist for jobs.

Additionally, we provide a text similarity analysis that matches jobs to resumes and vice versa.

## Results
After training several different models, a Support Vector Machine (SVM) model ended up performing the best with an accuracy
score of ~79%.

For the future, we would like to build a deep neural network. However, additional data needs to be scraped from Craigslist
before this would be plausible.

## Repository Organization
```/scripts``` contains three scripts and one notebook for scraping, loading data, and building models.

```/pck``` holds the pickled data scraped from the website.

```/csv``` contains the data used for the model as well as the output data from the model predictions and matching jobs to 
potential resumes and vice versa.

```/doc``` contains submission information (i.e. final presentation and report documents)

## Group Members
Ryan Egbert

Zainab Aljaroudi

Puja Gupta

Gagan Pahuja

Mrugshitr Bansal

