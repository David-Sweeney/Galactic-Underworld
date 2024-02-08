import numpy as np

class NatalKick:
    def __init__(self, distributions={'Black Hole': 'scaled igoshev young', 'Neutron Star': 'igoshev young'}):
        """
        Class for generating natal kicks from a distribution.
        
        Parameters
        ----------
        distributions : dict, optional
            The distributions to use for black holes and neutron stars. 
            The default is {'Black Hole': 'scaled igoshev young', 'Neutron Star': 'igoshev young'}.
            Options:
                'igoshev all', 'igoshev young', 'renzo', 'hobbs'
                'scaled igoshev all', 'scaled igoshev young', 'scaled renzo', 'scaled hobbs'                
        """
        
        self.string_representation = str(distributions)
        
        assert isinstance(distributions, dict), 'Distributions must be a dictionary'
        for species, distribution in distributions.items():
            assert isinstance(distribution, (int, float, str)) or callable(distribution), \
                f'Invalid distribution for {species}'
            if isinstance(distribution, str):
                if distribution.split()[0] == 'scaled':
                    assert ' '.join(distribution.split()[1:]) in ['igoshev all', 'igoshev young', 'renzo', 'hobbs'], \
                        f'Invalid distribution for {species}'
                else:
                    assert distribution in ['igoshev all', 'igoshev young', 'renzo', 'hobbs'], \
                        f'Invalid distribution for {species}'
                
                # Get the PDF function for the natal kick
                distributions[species] = self._get_PDF(distribution)
                
            
        self.distributions = distributions
        
        # Max velocity and max PDF are only used for inbuilt PDFs
        self.max_velocity = {species: 1500 for species in self.distributions}
        # Max value of PDF will be calculated as required
        self.max_pdf = {}
        
    def calculate_max_pdf(self, PDF, *args):
        """Calculates the maximum value of the PDF"""
        
        xs = np.linspace(0, self.max_velocity, 1000)
        ys = PDF(xs, *args)

        largest_prob_index = ys.argmax()
        focused_xs = np.linspace(xs[largest_prob_index-1], xs[largest_prob_index+1], 1000)
        max_pdf = PDF(focused_xs, *args).max()
        
        return max_pdf
        
    def get_kick(self, species, object_mass, progenitor_mass):
        """
        Returns an (x, y, z) kick velocity from the species distribution.
        
        Parameters
        ----------
        species : str
            The species of the object. Typical options: 'Black Hole', 'Neutron Star'.
        object_mass : float
            Mass of the object in solar masses.
        progenitor_mass : float
            Mass of the progenitor in solar masses.
        
        Returns
        -------
        x : float
            x-component of kick velocity in km/s.
        y : float
            y-component of kick velocity in km/s.
        z : float
            z-component of kick velocity in km/s.        
        """
        
        natal_kick = None
        elif isinstance(self.distributions[species], (int, float)):
            natal_kick = self.distributions[species]
        elif callable(self.distributions[species]):
            natal_kick = self.distributions[species](species, object_mass, progenitor_mass)
        elif isinstance(self.distributions[species], str):
            if species not in self.max_pdf:
                self.max_pdf[species] = self.calculate_max_pdf(self.distributions[species], 
                                                               species, object_mass, progenitor_mass)
                
            # Sample from inbuilt PDF
            while natal_kick is None:
                v = np.random.uniform(0, self.max_velocity[species])
                p = np.random.uniform(0, self.max_pdf[species])

                if p < self.distributions[species](v, object_mass, progenitor_mass):
                    natal_kick = v
        else:
            raise ValueError(f'Invalid distribution for {species}')

        # Convert natal kick into (x, y, z) using the Muller method
        u, v, w = np.random.normal(0, 1, size=3)
        norm = np.sqrt(u**2 + v**2 + w**2)
        x, y, z = natal_kick * np.array([u, v, w]) / norm
    
        return x, y, z
    
    def __str__(self):
        return self.string_representation
            

def maxwellian(v, sigma):
    """Maxwellian PDF"""
    return np.sqrt(2/np.pi) * v**2/sigma**3 * np.exp(-v**2/(2*sigma**2))

def _get_PDF(name):
    """
    Returns the PDF function for the provided distribution name.
    
    Parameters
    ----------
    name : str
        Name of the distribution. Options:
            'igoshev all', 'igoshev young', 'renzo', 'hobbs'
            'scaled igoshev all', 'scaled igoshev young', 'scaled renzo', 'scaled hobbs'
    
    Returns
    -------
    function
        The PDF function for the distribution.
    """
    
        def PDF(v, object_mass, progenitor_mass):
        """
        Returns the PDF value of the provided kick velocity as evaluated
        for a particular object and progenitor mass.
        
        Parameters
        ----------
        v : float
            Velocity in km/s.
        object_mass : float
            Mass of the object in solar masses.
        progenitor_mass : float
            Mass of the progenitor in solar masses.
        
        Returns
        -------
        float
            The PDF value at the provided kick velocity.
        """
        
        if progenitor_mass >= 40:
            return 0
        
        value = w * self.maxwellian(v, sigma_1) + (1 - w) * self.maxwellian(v, sigma_2)
        
        if rescale:
            value = (1.35 / object_mass) * value
        
        return value

    # Set the distribution to be rescaled if specified
    rescale = False
    if name.split()[0] == 'scaled':
        name = ' '.join(name.split()[1:])
        rescale = True
    
    # Igoshev (2020) main numbers
    if name == 'igoshev all':
        w = 0.42
        sigma_1 = 128
        sigma_2 = 298
    # Igoshev (2020) numbers for young pulsars
    elif name == 'igoshev young':
        w = 0.20
        sigma_1 = 56
        sigma_2 = 336
    elif name == 'renzo':
        w = 0.02
        sigma_1 = 1
        sigma_2 = 16
    elif name == 'hobbs':
        w = 0
        sigma_1 = 0
        sigma_2 = 265
    else:
        raise ValueError(f'Unknown distribution: {distribution}')
    
    return PDF