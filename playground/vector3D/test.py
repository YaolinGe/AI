import plotly.express as px
import plotly.io as pio

# Set MathJax to None if necessary
# pio.kaleido.scope.mathjax = None

# Create a figure
fig = px.line(x=[0,1,2], y=[0,1,2])

# Export the figure
fig.write_image("plot.png", format='png')
