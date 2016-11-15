from astropy.modeling.models import custom_model
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt

# Philip Mocz (2016)
# Harvard University
# Astropy fitting tutorial
# A simple 1D example

np.random.seed(42)

# Define a sinusoidal and linear model functions as a custom 1D model
def sine_model(x, amplitude=1., frequency=1.):
  return amplitude * np.sin(2 * np.pi * frequency * x)

def linear_model(x, m=1., b=0.):
  return m * x + b

SineModel = custom_model(sine_model)
LinearModel =  custom_model(linear_model)
 
# Create an instance of the custom model and evaluate it
model_init = SineModel(amplitude=0.75, frequency=2.3) + LinearModel()
print("f_init(0.25)=", model_init(0.25),"\n")


# Create mock data with errorbars
n_data = 256
amplitude_exact = 0.7
frequency_exact = 2.4
m_exact = 0.1
b_exact = 0.2
error_amplitude = 0.04
x_data = np.linspace(0, 4, n_data)
y_data = amplitude_exact * np.sin(2 * np.pi * frequency_exact * x_data) + m_exact*x_data + b_exact + error_amplitude * np.random.randn(n_data) 
y_error = error_amplitude * np.random.randn(n_data) # errorbars of measurements


# Fit the data using the astropy custom model
fit_mymodel = fitting.LevMarLSQFitter()
f_astropy = fit_mymodel(model_init, x_data, y_data)


# Print results
print("exact: [", amplitude_exact, frequency_exact, m_exact, b_exact, "]\n")
print("fit: [", f_astropy.amplitude_0.value, f_astropy.frequency_0.value, f_astropy.m_1.value, f_astropy.b_1.value, "]\n")


# Make  figure
plt.plot(x_data, y_data, label=r'$\rm{data}$', linestyle='none', marker='o', color='r', markersize=12, markeredgewidth=1)
plt.plot(x_data, f_astropy(x_data), label=r'$\rm{astropy\,fit}$', linewidth=2)
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$y$', fontsize=20)
plt.legend(loc='lower right')
plt.savefig("astropyFit.pdf")
plt.show()
