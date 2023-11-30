# neuroNLP dashboard manual

## Installation

Requirements:

- A  Linux OS computer with docker installed (version 20 or later).

The neuroNLP software is distributed as a docker image, as such it requires docker to run. The docker image can be acquired either from the docker registry or as a tarball (a .tar file). 

1. If the image is supplied as a tarball, unpack the tarball using `docker load < imagename.tar `. This will add the image to the local docker registry. 
2. Check the image has been successfully registered with `docker image ls`. You should see a new entry with the name of the dashboard image.
3. Start the dashboard container using the run command `docker container run -p 8501:8501 imagename ` . This will spin up a locally running container of the neuroNLP software. 
4. The dashboard runs on a local network, and is accessed via a browser. To check the container is running successfully, check the address `localhost:8501`. The browser should display something like 

![Screenshot 2022-05-31 at 13-41-59 app · Streamlit](/home/hwatkins/Desktop/neuroNLP/dashboard_assets/Screenshot 2022-05-31 at 13-41-59 app · Streamlit.png)

The AI models required by the software will take a a couple of minutes to load up. Afterwards, one should see

![Screenshot 2022-05-31 at 13-42-32 app · Streamlit](/home/hwatkins/Desktop/neuroNLP/dashboard_assets/Screenshot 2022-05-31 at 13-42-32 app · Streamlit.png)

The dashboard is now ready to use.

## Usage

### Uploading data
To use the dashboard, one must first upload some data you wish to analyse. Currently only CSV and XLSX files are supported. Drag-and-drop into the bar, or use the "Browse files" tab o upload your data. If your excel spreadsheet is password-protected, a box will appear, prompting you to provide your file password. Fill in your password and press enter, the press the 'Upload and process reports' button to complete the upload. 

The dashboard also supports uploading multiple files. When you want to upload several files, simply drag and drop all your files into the bar, the dashboard will read in all these files separately then concatenate the data together.

### Selecting reports

Once the data has been loaded and processed, the dashboard will display a dropdown menu with selection options.

![Screenshot 2022-10-18 at 14-00-12 app · Streamlit](/home/hwatkins/Desktop/neuroNLP/dashboard_assets/Screenshot 2022-10-18 at 14-00-12 app · Streamlit.png)

You can use these tabs to specify criteria for report selection. You can choose a particular date range, age range, sex, department etc. The following plots and analysis will only consider the subset of reports that correspond to the specified criteria.

### Operational view

The dashboard has two "views", which can be toggled using the drop-down on the left-hand-side of the screen. The "operational" view provides statistics for a large number of reports, and plotting functionality to view the data. To plot a particular variable, select one from the 'variable to plot' dropdown and choose a plot view. These plot views allow you to create temporal or integrated plots, or to change the axis to a logarithmic scale. These plots are interactive! Try clicking a dragging, zooming in or displaying information by passing the mouse over the plot.

![Screenshot 2022-10-18 at 14-04-09 app · Streamlit](/home/hwatkins/Desktop/neuroNLP/dashboard_assets/Screenshot 2022-10-18 at 14-04-09 app · Streamlit.png)

The operational view displays a summary panel below the plot that contains a statistical summary of the selected data.

![Screenshot 2022-10-18 at 14-08-08 app · Streamlit](/home/hwatkins/Desktop/neuroNLP/dashboard_assets/Screenshot 2022-10-18 at 14-08-08 app · Streamlit.png)

One can also export the results of the plotting and statistical summary using the "export selection as csv" and "export analysis as pdf" tabs. The "export selection as csv" button will download your selected reports in a CSV format. The "export analysis as pdf" button will download a pdf copy of the plot and summary created during the session.

### Clinical view

The "clinical" view shows details of individual reports, including the results of machine-learning text recognition and clinical entity detection pipelines. The clinical view displays individual reports and associated metadata, including the AI-informed clinical concept detection. Select a subset of reports of interest then specify which particular report you wish to view using the "Patient ID" and "Report Row to select" tabs.

![Screenshot 2022-10-18 at 14-13-53 app · Streamlit](/home/hwatkins/Desktop/neuroNLP/dashboard_assets/Screenshot 2022-10-18 at 14-13-53 app · Streamlit.png)
