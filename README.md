# A/B Test Statistical Significance Analyzer

## Overview

This is a Streamlit-based web application designed to analyze the statistical significance of A/B test results. The application provides a user-friendly interface for inputting test data and generates comprehensive statistical analysis with actionable insights. It's built as a single-page application using Python and focuses on statistical analysis and data visualization.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - chosen for its simplicity in creating data science web applications
- **Layout**: Wide layout with sidebar configuration and main content area split into columns
- **Components**: Interactive widgets for user input, data visualization charts, and results display
- **Styling**: Uses Streamlit's built-in styling with custom page configuration

### Backend Architecture
- **Language**: Python
- **Architecture Pattern**: Single-file application with functional programming approach
- **Processing**: Real-time statistical calculations performed on user input
- **No traditional backend**: All processing happens client-side through Streamlit's server

### Data Processing
- **Statistical Library**: SciPy for statistical tests and calculations
- **Data Manipulation**: Pandas and NumPy for data handling and numerical operations
- **Mathematical Operations**: Built-in Python math functions for basic calculations

## Key Components

### 1. User Interface Components
- **Sidebar Configuration**: Test parameters including significance level, test type, and input method
- **Main Content Area**: Split into two columns for balanced layout
- **Interactive Widgets**: Selectboxes, radio buttons, and input fields for user interaction

### 2. Statistical Analysis Engine
- **Significance Testing**: Implements various statistical tests for A/B testing
- **Test Types**: Supports two-tailed and one-tailed hypothesis testing
- **Configurable Parameters**: Adjustable significance levels (0.05, 0.01, 0.001)

### 3. Data Visualization
- **Charting Library**: Plotly for interactive visualizations
- **Chart Types**: Support for various graph objects and express charts
- **Real-time Updates**: Charts update dynamically based on user input

### 4. Input Handling
- **Flexible Input Methods**: 
  - Conversion rates (percentage-based input)
  - Raw counts (absolute numbers)
- **Data Validation**: Built-in validation through Streamlit widgets

## Data Flow

1. **User Input**: Users select test configuration and input method through sidebar
2. **Data Entry**: Test data entered through main interface (conversion rates or raw counts)
3. **Statistical Processing**: Real-time calculation of statistical significance using SciPy
4. **Results Generation**: Comprehensive analysis including p-values, confidence intervals, and effect sizes
5. **Visualization**: Interactive charts generated using Plotly
6. **Actionable Insights**: Interpreted results presented to user

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scipy**: Scientific computing and statistical functions
- **plotly**: Interactive data visualization
- **math**: Mathematical functions (built-in Python)

### Why These Dependencies Were Chosen
- **Streamlit**: Rapid prototyping and deployment of data science applications
- **SciPy**: Industry-standard statistical computing library
- **Plotly**: Interactive visualizations superior to static charts for data exploration
- **Pandas/NumPy**: Standard data science stack for Python

## Deployment Strategy

### Current Setup
- **Platform**: Designed for Replit deployment
- **Single File**: Entire application contained in `app.py` for simplicity
- **No Database**: Stateless application with no persistent data storage
- **No Authentication**: Open access application

### Deployment Considerations
- **Scalability**: Streamlit handles concurrent users through session state
- **Performance**: Lightweight application with minimal resource requirements
- **Maintenance**: Single-file architecture makes updates and debugging straightforward

### Future Enhancements
- **Data Persistence**: Could add database integration for saving test results
- **User Authentication**: Could implement user accounts for saving multiple tests
- **Advanced Analytics**: Could expand statistical tests and analysis methods
- **Export Functionality**: Could add PDF/CSV export capabilities

## Technical Decisions

### Problem Solved
Provides an accessible tool for non-statisticians to perform rigorous A/B test analysis without requiring statistical software or programming knowledge.

### Key Design Choices
1. **Streamlit over Flask/Django**: Faster development and better suited for data science applications
2. **Single-file architecture**: Simplifies deployment and maintenance
3. **Real-time processing**: Immediate feedback enhances user experience
4. **Multiple input methods**: Accommodates different data collection approaches
5. **Interactive visualizations**: Plotly chosen over matplotlib for better user engagement

### Trade-offs
- **Pros**: Rapid development, easy deployment, user-friendly interface
- **Cons**: Limited customization compared to full web frameworks, single-user session limitations

### Preview:
<img width="1919" height="972" alt="image" src="https://github.com/user-attachments/assets/a9c8fada-68b0-4033-9152-028ec98d926c" />
<img width="1919" height="965" alt="image" src="https://github.com/user-attachments/assets/1dcb17c3-fb92-4cc2-a1a8-0108fded3513" />
<img width="1919" height="965" alt="image" src="https://github.com/user-attachments/assets/79930229-aea0-49c8-b7af-409e3959d4cb" />
<img width="1919" height="957" alt="image" src="https://github.com/user-attachments/assets/5873110e-50c4-4dfa-91d5-d96006160370" />
<img width="1919" height="446" alt="image" src="https://github.com/user-attachments/assets/c07603ef-1dc4-4599-a178-5b5301596d4f" />
