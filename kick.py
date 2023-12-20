import numpy as np

class NatalKick:
    def __init__(self, distribution='igoshev_young', bh_kicks='Scaled'):
        """
        Class for generating natal kicks from a distribution.
        
        Parameters
        ----------
        distribution : str, optional
            The distribution to use. The default is 'igoshev_young'.
        bh_kicks : str, optional
            How to handle black hole kicks. The default is 'Scaled'.
            Options:
                'Zero': Set black hole kicks to zero.
                'Scaled': Scale black hole kicks to have the same momentum as neutron star kicks.
                'Equal': Set black hole kicks to have the same velocity as neutron star kicks.
        """
        
        assert distribution in ['igoshev_all', 'igoshev_young', 'renzo', 'hobbs'], \
            f'Unknown distribution: {distribution}'
        assert bh_kicks in ['Zero', 'Scaled', 'Equal'], \
            f'Unknown bh_kicks option: {bh_kicks}'
            
        self.distribution = distribution
        self.bh_kicks = bh_kicks
        
        # Maximum velocity in km/s to sample from the PDF
        self.max_velocity = 1500
        
        # Maximum value of the PDF
        self.max_pdf = self.calculate_max_pdf()
    
    def maxwellian(self, v, sigma):
        """Maxwellian PDF"""
        return np.sqrt(2/np.pi) * v**2/sigma**3 * np.exp(-v**2/(2*sigma**2))
    
    def PDF(self, v, black_hole=False, ECSN=False):
        """
        Returns the PDF value of the provided kick velocity as evaluated 
        by the class' distribution.
        
        If the kick is for a black hole, it is never from an ECSN. ECSN kicks 
        are attributed to the smaller peak (sigma_1) of the distribution, so 
        for a black hole the PDF is evaluated using only the larger peak (sigma_2).
        
        Parameters
        ----------
        v : float
            Velocity in km/s.
        black_hole : bool, optional
            Whether the kick is for a black hole. The default is False.
        ECSN : bool, optional
            Whether the kick is from an electron capture supernova (ECSN). 
            The default is False.
        
        Returns
        -------
        float
            The PDF value at the provided kick velocity as evaluated by the class' distribution.
        """
        # Igoshev (2020) main numbers
        if self.distribution == 'igoshev_all':
            w = 0.42
            sigma_1 = 128
            sigma_2 = 298
        # Igoshev (2020) numbers for young pulsars
        elif self.distribution == 'igoshev_young':
            w = 0.20
            sigma_1 = 56
            sigma_2 = 336
        elif self.distribution == 'renzo':
            w = 0.02
            sigma_1 = 1
            sigma_2 = 16
        elif self.distribution == 'hobbs':
            w = 0
            sigma_1 = 0
            sigma_2 = 265
        else:
            raise ValueError(f'Unknown distribution: {distribution}')

        # If the kick is for a black hole, it is never from an ECSN (i.e. the sigma_1 peak)
        if black_hole:
            return self.maxwellian(v, sigma_2)
        # Return the PDF value of the ECSN peak
        if ECSN:
            return w*self.maxwellian(v, sigma_1)
        return w * self.maxwellian(v, sigma_1) + (1 - w) * self.maxwellian(v, sigma_2)
    
    def calculate_max_pdf(self):
        """Calculates the maximum value of the PDF"""

        xs = np.linspace(0, self.max_velocity, 1000)
        ys = self.PDF(xs, black_hole=False)
        y_bhs = self.PDF(xs, black_hole=True)

        if ys.max() > y_bhs.max():
            largest_prob_index = ys.argmax()
            black_hole = False
        else:
            largest_prob_index = y_bhs.argmax()
            black_hole = True
            
        focused_xs = np.linspace(xs[largest_prob_index-1], xs[largest_prob_index+1], 1000)
        max_pdf = self.PDF(focused_xs, black_hole=black_hole).max()
        # print(f'Maximum value of PDF: {MAX_PDF:.5f}')
        return max_pdf
    
    def determine_ECSN(self, v, black_hole=False):
        '''Returns True if the kick is from an ECSN'''
        # BHs can't be formed by ECSN
        if black_hole:
            return False
        
        chance_ECSN = self.PDF(v, ECSN=True) / self.PDF(v)
        if np.random.uniform() < chance_ECSN:
            return True
        else:
            return False
    
    def get_kick(self, black_hole=False, heavy=False):
        """
        Returns an (x, y, z) kick velocity from the distribution and whether 
        the kick comes from an electron capture supernova (ECSN).
        
        If the kick is for a black hole, it is never from an ECSN. Black hole kicks 
        can be set to zero by setting the bh_kicks class parameter to 'Zero', scaled 
        to have the same momentum as a NS kick (assuming NS mass is 1.35 Msun and BH 
        mass is 7.8 Msun) by setting bh_kicks to 'Scaled', or have equal velocities 
        to NS kicks by setting bh_kicks to 'Equal'.
        
        If the kick is for a heavy star (M > 40 Msun), the kick is set to 0.
        
        Parameters
        ----------
        black_hole : bool, optional
            Whether the kick is for a black hole. The default is False.
        heavy : bool, optional
            Whether the kick is for a heavy star (M > 40 Msun). The default is False.
        
        Returns
        -------
        x : float
            x-component of kick velocity in km/s.
        y : float
            y-component of kick velocity in km/s.
        z : float
            z-component of kick velocity in km/s.
        ECSN : bool
            Whether the kick is from an electron capture supernova (ECSN).
        
        """
        
        # Get natal kick from PDF
        if black_hole and (self.bh_kicks == 'Zero' or heavy):
            natal_kick = 0
        else:
            natal_kick = None
            
        while natal_kick is None:
            v = np.random.uniform(0, self.max_velocity)
            p = np.random.uniform(0, self.max_pdf)

            if p < self.PDF(v):
                natal_kick = v
        
        ECSN = self.determine_ECSN(natal_kick, black_hole)
        
        # Scale BH kicks to have same momentum as NS kicks
        if self.bh_kicks == 'Scaled':
            # Assumes NS mass is 1.35 Msun and BH mass is 7.8 Msun
            natal_kick *= 0.173

        # Convert natal kick into (x, y, z) using the Muller method
        u, v, w = np.random.normal(0, 1, size=3)
        norm = np.sqrt(u**2 + v**2 + w**2)
        x, y, z = natal_kick * np.array([u, v, w]) / norm
    
        return x, y, z, ECSN
            
    