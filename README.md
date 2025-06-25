Python script for automating placing FCs
citypoints.shp is a point file of settlement centroids
frontline.shp is a line file of the frontline as of 6/25/2025
Please upload a CSV with the name citiesMMDD to the function. This should have a single column with "FC" as the header and the list of settlements following it.
Update the function call in line 135 to reflect the date
This will export a .shp of points for FCs and a corresponding CSV with lat/long data
