# CONVERSIONS
um_2_m = 1E-6
mm_2_m = 1E-3
m_2_mm = 1E3
m_2_um = 1E6
s_2_ms = 1E3
Pa_2_bar = 1E-5
uL_2_mL = 1E-3
min_2_s = 60
s_2_min = 1/60
uLmin_2_m3s = 1/60E9

# conversion from pixels to microns (camera and magnification-dependent)
pix_per_um = {'chronos' : 
                {4 : 1.34, # measured based on reference dot for 4x objective
                10 : 3.54}, # see get_pixel_width_from_calibration_slide.py
            'photron' :
                {4: 1/2.29, 
                10 : 1.09,
                20 : 2.18},
}
