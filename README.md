# Diamond Pricing Analysis and Prediction
![Diamond Header](images/diamond_clarity_analysis.png)

## Introduction
_As a Data Analyst, my objective in this project is to assist an online jewelry retailer in developing a more efficient approach to pricing newly acquired diamonds. With thousands of diamonds requiring pricing each day, the company is seeking a cost-effective and scalable pricing model that can support experts in making fast and consistent pricing decisions._

_To achieve this, I am working with a [dataset of historical diamond sales](https://www.kaggle.com/datasets/shivam2503/diamonds) to uncover pricing patterns and build a predictive model. The goal is to estimate prices accurately based on key diamond features such as carat, cut, color, and clarity. By automating the initial price estimation process, this model enables pricing specialists to work more efficiently&ndash;reducing manual effort while maintaining accuracy and speed in daily operations._

_Diamonds are evaluated based on a set of core characteristics known as the four **C**'s: **_cut_**, **_color_**, **_clarity_**, and **_carat_**. **_Cut_** refers to the quality of a diamond's form. Well-cut diamonds are symmetrical and reflect light well, giving them a sparkly appearance. **_Color_** is the color of the stone. The clearer the diamond, the higher its color grade. Yellower diamonds are less valuable. **_Clarity_** refers to the number of imperfections on the surface of the stone or within it. Clearer diamonds are more valuable. **_Carat_** is a measure of weight used for gems. A one-carat round-cut diamond is about the size of a green pea. The term carat actually comes from the carob seed, which was historically used to weigh precious stones._

The features in the dataset include:

| Column    | Description                                                                         |
| --------- | ----------------------------------------------------------------------------------- |
| `carat`   | Weight of the diamond (in **metric carats**, where 1 carat = 0.2 grams)             |
| `cut`     | Quality of the cut (categorical: **Fair, Good, Very Good, Premium, Ideal**)         |
| `color`   | Diamond color (categorical: from **D** (best) to **J** (worst))                     |
| `clarity` | Clarity of the diamond (categorical: from **IF** (flawless) to **I1** (inclusions)) |
| `depth`   | Total depth percentage = `z / mean(x, y)` (as a % of width)                         |
| `table`   | Width of the top of the diamond relative to its widest point                        |
| `price`   | Price in US dollars                                                                 |
| `x`       | Length of the diamond when viewed face-up                                           |  
| `y`       | Width of the diamond when viewed face-up                                            |
| `z`       | Height of the diamond when standing on its point.                                   |
