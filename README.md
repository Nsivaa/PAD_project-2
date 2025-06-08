# PAD_project-2

Second part (text mining) of the Project for the Data Analysis and Mining course at NOVA Universidade de Lisboa, 2024-2025.  

## Running
To run the Jupyter Notebook, it is required for the appropriate text corpus data to be located in the `data/corpus2mw/` folder. 
The cohesion metric can be chosen between "scp", "phi_square" and "dice", and can be selected by manually modifying the string parameter in the construction of the Extractor object, in the first cell of the notebook. The default is "scp". 

## Requirements
The required packages are listed in the `requirements.txt` file and can be installed via `pip install -r /path/to/requirements.txt`.