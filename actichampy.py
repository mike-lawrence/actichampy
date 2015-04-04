#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# This code includes modifications of the Vispy example at:
# https://github.com/vispy/vispy/tree/5098146746c098c7af9e4e09ea44ff6ef5ea2187/examples/demo/gloo/realtime_signals.py
# See the file "LICENSE_vispy.txt" for the copyright license associated with that original example.
#
# Otherwise, modifications made herein are considered copyright under the terms described in the file "LICENSE_actichampy.txt"


from vispy import gloo
from vispy import app
import numpy as np
import scipy.fftpack
import scipy.special
import math
import OpenGL.GL as gl
import time

VERT_SHADER = """
#version 120

// y coordinate of the position.
attribute float a_position;

// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;

// Size of the table.
uniform float u_nrows;

// Number of samples per signal.
uniform float u_n;

// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;

void main() {

	// Compute the x coordinate from the time index.
	float x = -1 + 2*a_index.z / (u_n-1);
	vec2 position = vec2(x, a_position);

	// Find the affine transformation for the subplots.
	vec2 a = vec2(1., 1./u_nrows)*.9;
	vec2 b = vec2( -1 + 2 * (a_index.x+.5) ,
				   -1 + 2 * (a_index.y+.5) / u_nrows );

	// Apply the static subplot transformation + scaling.
	gl_Position = vec4(a*position+b, 0.0, 1.0);

	v_index = a_index;

	// For clipping test in the fragment shader.
	v_position = gl_Position.xy;
	v_ab = vec4(a, b);
}
"""

FRAG_SHADER = """
#version 120

// Number of samples per signal.
uniform float u_alpha;

varying vec3 v_index;

varying vec2 v_position;
varying vec4 v_ab;

void main() {
	gl_FragColor = vec4(1.,1.,1.,u_alpha);

	// Discard the fragments between the signals (emulate glMultiDrawArrays).
	if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
		discard;

	// Clipping test.
	vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
	if ((test.x > 1) || (test.y > 1))
		discard;
}
"""


class Canvas(app.Canvas):
	def update_index(self):
		self.index = np.c_[np.repeat(np.repeat(0, self.nrows), 2**self.samples_per_screen_pow),
			  np.repeat(np.tile(np.arange(self.nrows), 1), 2**self.samples_per_screen_pow),
			  np.tile(np.arange(2**self.samples_per_screen_pow), self.nrows)].astype(np.float32)
	def __init__(self,data):
		self.data = data
		self.alpha_shift = 12
		self.samples_per_screen_pow = np.floor(np.log2(self.data.shape[1]))
		self.show_latest = True
		self.last_sample_to_show = self.data.shape[1]#None
		self.nrows = self.data.shape[0]
		self.update_index() #creates self.index
		self.equalize_amplitudes = False
		self.bandpass = False
		app.Canvas.__init__(self, title='Use your wheel to zoom!',keys='interactive')
		self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
		self.program['a_position'] = self.data.reshape(-1, 1)
		self.program['a_index'] = self.index
		self.program['u_nrows'] = self.nrows
		self.program['u_n'] = 2**self.samples_per_screen_pow
		self.program['u_alpha'] = 1-scipy.special.expit((self.samples_per_screen_pow-self.alpha_shift))#1.0/np.log2(self.samples_per_screen_pow**2)
		gloo.set_state(clear_color='black', blend=True,
					blend_func=('src_alpha', 'one_minus_src_alpha'))
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		gl.glEnable( gl.GL_LINE_SMOOTH )
		gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST )

	def on_resize(self, event):
		self.width, self.height = event.size
		gloo.set_viewport(0, 0, self.width, self.height)
	def on_key_press(self, event):
		if event.key.name=='Space': #toggle show_latest on/off
			self.show_latest = not self.show_latest
		elif event.key.name=='A':
			self.equalize_amplitudes = not self.equalize_amplitudes
		elif event.key.name=='B':
			self.bandpass = not self.bandpass
		elif event.key.name=='Up':
			self.samples_per_screen_pow += 1
		elif event.key.name=='Down':
			self.samples_per_screen_pow -= 1			
		elif event.key.name=='Left':
			self.alpha_shift += 1
		elif event.key.name=='Right':
			self.alpha_shift -= 1			
		else:
			print event.key.name
	def on_mouse_wheel(self, event):
		if not self.show_latest:
			self.last_sample_to_show = self.last_sample_to_show + int((2**self.samples_per_screen_pow)*(event.delta[0]/100))
			if self.last_sample_to_show < (2**self.samples_per_screen_pow):
				self.last_sample_to_show = (2**self.samples_per_screen_pow)
			elif self.last_sample_to_show > self.data.shape[1]:
				self.last_sample_to_show = self.data.shape[1]
	def on_draw(self, event):
		start = time.time()
		"""Add some data at the end of each signal (real-time signals)."""
		k = int(1000/60) #emulating 1kHz new data rate (assuming this function gets called every refresh at 60Hz refresh rate)
		self.data = np.c_[self.data,amplitudes * np.random.randn(self.nrows, k)]
		if self.samples_per_screen_pow<2:
			self.samples_per_screen_pow = 2
		while (2**self.samples_per_screen_pow)>self.data.shape[1]:
			self.samples_per_screen_pow -= 1
		if self.show_latest:
			self.last_sample_to_show = self.data.shape[1]
			this_data = self.data[:,-(2**self.samples_per_screen_pow):].copy()
		else:
			while (2**self.samples_per_screen_pow)>self.last_sample_to_show:
				self.samples_per_screen_pow -= 1
			this_data = self.data[:,self.last_sample_to_show-(2**self.samples_per_screen_pow):self.last_sample_to_show].copy()
		from_fft = scipy.fftpack.rfft(this_data)
		# power = 2.0/from_fft.shape[1] * np.abs(from_fft[0:from_fft.shape[1]/2])
		if self.bandpass:
			#apply bandpass
			W = scipy.fftpack.fftfreq(this_data.shape[1], d=0.001)
			from_fft[:,(W<0.1)] = 0
			from_fft[:,(W>30)] = 0
			this_data = scipy.fftpack.irfft(from_fft)
		if self.equalize_amplitudes:
			this_data = this_data.transpose()
			this_data -= np.amin(this_data,axis=0)
			this_data /= np.amax(this_data,axis=0)
			this_data *= 2
			this_data -= 1
			this_data = this_data.transpose()
		else:
			this_data -= np.amin(this_data)
			this_data /= np.amax(this_data)
			this_data *= 2
			this_data -= 1
		# print time.time()-start
		self.update_index()
		self.program['a_index'] = self.index
		self.program['u_n'] = 2**self.samples_per_screen_pow
		self.program['u_alpha'] = 1-scipy.special.expit((self.samples_per_screen_pow-self.alpha_shift))#1.0/np.log2(self.samples_per_screen_pow**2)
		self.program['a_position'] = this_data.reshape(-1, 1).astype(np.float32)
		self.update()
		gloo.clear()
		self.program.draw('line_strip')


if __name__ == '__main__':
	nrows = 32 
	n = 10000
	amplitudes = .1 + .2 * np.random.rand(nrows, 1).astype(np.float32)
	y = amplitudes * np.random.randn(nrows, n).astype(np.float32)
	c = Canvas(data=y)
	c.show()
	app.run()
