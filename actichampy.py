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

varying vec3 v_index;

varying vec2 v_position;
varying vec4 v_ab;

void main() {
	gl_FragColor = vec4(1.,1.,1.,0.9);

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
		self.index = np.c_[np.repeat(np.repeat(0, self.nrows), self.samples_per_screen),
			  np.repeat(np.tile(np.arange(self.nrows), 1), self.samples_per_screen),
			  np.tile(np.arange(self.samples_per_screen), self.nrows)].astype(np.float32)
	def __init__(self,data):
		self.data = data
		self.samples_per_screen = self.data.shape[1]
		self.show_latest = True
		self.scroll_time = False
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
		self.program['u_n'] = self.samples_per_screen
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
			self.last_sample_to_show = self.data.shape[1]
		elif event.key.name=='Z': #toggle scoll_time on
			self.scroll_time = True
		elif event.key.name=='A':
			self.equalize_amplitudes = not self.equalize_amplitudes
		elif event.key.name=='B':
			self.bandpass = not self.bandpass
		else:
			print event.key.name
	def on_key_release(self, event):
		if event.key.name=='Z': #toggle scoll_time back off
			self.scroll_time = False
	def on_mouse_wheel(self, event):
		if not self.scroll_time:
			dy = np.sign(event.delta[1]) * .05
			self.samples_per_screen = int(self.samples_per_screen * math.exp(2.5*dy))
			if self.samples_per_screen<100:
				self.samples_per_screen = 100
			elif self.samples_per_screen>self.data.shape[1]:
				self.samples_per_screen = self.data.shape[1]
		elif not self.show_latest:
			self.last_sample_to_show = self.last_sample_to_show + int(self.samples_per_screen*(event.delta[0]/100))
			if self.last_sample_to_show < self.samples_per_screen:
				self.last_sample_to_show = self.samples_per_screen
			elif self.last_sample_to_show > self.data.shape[1]:
				self.last_sample_to_show = self.data.shape[1]
	def on_draw(self, event):
		"""Add some data at the end of each signal (real-time signals)."""
		k = int(1000/60) #emulating 1kHz new data rate (assuming this function gets called every refresh at 60Hz refresh rate)
		self.data = np.c_[self.data,amplitudes * np.random.randn(self.nrows, k)]
		next2pow = np.ceil(np.log2(self.samples_per_screen))
		trim_start = None
		trim_end = None
		trim_needed = False
		if self.show_latest:
			self.last_sample_to_show = self.data.shape[1]
			try2pow = next2pow
			done = False
			while not done:
				if self.data.shape[1]>=(2**try2pow): #plenty of data to use and trim
					this_data = self.data[:,-(2**try2pow):].copy()
					done = True
					if self.data.shape[1]>(2**try2pow): #need to trim
						trim_needed = True
						trim_start = this_data.shape[1]-self.samples_per_screen
						trim_end = this_data.shape[1]
				else: #can't show samples_per_screen, try next power down
					try2pow -= 1
					if (2**try2pow)<self.samples_per_screen: #decrease samples_per_screen is necessary
						self.samples_per_screen = 2**try2pow
		else: #not showing latest
			try2pow = next2pow
			done = False
			while not done: #find a good widow size
				if (2**try2pow)<self.samples_per_screen: #decrease samples_per_screen is necessary
					self.samples_per_screen = 2**try2pow
				window_room = self.last_sample_to_show - (2**try2pow)
				if window_room>=0: #enough room before last sample
					done = True
					this_data = self.data[:,window_room:self.last_sample_to_show].copy()
					if (2**try2pow)>self.samples_per_screen:
						trim_needed = True
						trim_start = this_data.shape[1]-self.samples_per_screen
						trim_end = this_data.shape[1]
				else: #window_room<0, not enough data before last sample
					if (self.last_sample_to_show-window_room)>self.data.shape[1]: #not enough data after the sample
						try2pow -= 1
					else: #enough room after last sample
						done = True
						this_data = self.data[:,0:(2**try2pow)].copy()
						if (2**try2pow)>self.samples_per_screen:
							trim_needed = True
							trim_start = self.last_sample_to_show - self.samples_per_screen
							trim_end = self.last_sample_to_show
		from_fft = scipy.fftpack.rfft(this_data)
		# power = 2.0/from_fft.shape[1] * np.abs(from_fft[0:from_fft.shape[1]/2])
		if self.bandpass:
			#apply bandpass
			W = scipy.fftpack.fftfreq(this_data.shape[1], d=0.001)
			from_fft[:,(W<0.1)] = 0
			from_fft[:,(W>30)] = 0
			this_data = scipy.fftpack.irfft(from_fft)
		if trim_needed:
			this_data = this_data[:,trim_start:trim_end]
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
		self.update_index()
		self.program['a_index'] = self.index
		self.program['u_n'] = self.samples_per_screen
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
