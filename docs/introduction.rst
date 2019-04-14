Introduction
============


Welcome to pyfor, a Python module intended for processing large aerial LiDAR (and phodar)
acquisitions for the use of large-scale forest inventories. This document serves as a general
introduction to the package, its philosophy, and a bit of its history. If you would like to
jump straight into analysis, feel free to skip over this "soft" document.

About the Author
----------------

I am Bryce Frank, a PhD student at Oregon State University. I work for the Forest Measurements
and Biometrics Lab, which is run under the guidance of Dr. Temesgen Hailemariam.
Our work focuses on developing statistical methodology for the analysis of forest resources
at multiple scales. Some of the lab members work on small-scale issues, like biomass
and taper modeling. My work, along with others, is focused on producing reliable
estimates of forest attributes for large scale forest assessment.

Package History
---------------

I originally began pyfor as a class project to explore the use of object oriented programming
in GIS. At the time, I had been programming in Python for about two years, but still struggled
with some concepts like Classes, object inheritance, and the like. pyfor was a way for me to
finally learn some of those concepts and implement them in an analysis environment that was
useful for me. Around the Spring of 2017, I released the package on GitHub. At the time,
the package was in very rough condition, was very inefficient, and only did a few rudimentary
tasks.

Around the Spring of 2018 I found a bit of time to rework the package from the ground up.
I was deeply inspired by the lidR package, which I used extensively for a few months.
I think lidR is a great tool, and pyfor is really just an alternative way of doing many of
the same tasks. However, I prefer to work in Python for many reasons, and I also prefer to
do my own scripting, so lidR fell by the wayside for me for those reasons. Rather than keep
my scripts locked up somewhere, I modified the early version of pyfor with my newest
attempts. I am also indebted to Bob McGaughey's FUSION, which paved the way in terms
of these sorts of software, and is still my go-to software package for production work.

Philosophy
----------

pyfor started as a means for me to learn OOP, and I think the framework is a very natural
way to work with LiDAR data from an interactive standpoint. In this way, pyfor is a bit
of a niche tool that is really designed more for research - being able to quickly change
parameters on the fly and get quick visual feedback about the results is important for tree
detection and other tasks. Because I am a bit selfish when I develop, and I am mainly a
researcher at this point in my career, this will be the main objective for the package
for the time being.

However, I completely understand the desire for performant processing. As the structure of
pyfor beings to solidify, more of my time can be spent on diagnosing performance issues
within the package and optimizing toward that end. I think Python, specifically scientific
Python packages, will continue to be a solid platform for developing reasonably performant
code, if done well. It is unlikely that pyfor will achieve speeds equivalent to raw C++ or
FORTRAN code, but I do not think it will be orders of magnitude away if functions are
developed in a way that leverages some of these faster packages. This also comes with
the benefit of increased maintainability, clarity and usability - goals that I feel
are often overlooked in the GIS world.

Acknowledgements
----------------

A special thank you to Drs. Ben Weinstein, Francisco Mauro-Gutiérrez and Temesgen Hailemariam
for their continued support and advice as this project matures.