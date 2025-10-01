#!/usr/bin/env python3
"""
Optimized geomeTRIC PySander in-memory API with constraints
This shows how to perform constrained optimization with zero I/O and optimized performance.
"""

import sys
import numpy as np
import tempfile
import os
import contextlib
from typing import Dict

from .molecule import Molecule
from .engine import PySander
from .internal import DelocalizedInternalCoordinates
from .optimize import Optimizer, OPT_STATE
from .params import OptParams
from .prepare import parse_constraints
from .nifty import ang2bohr, bohr2ang

# Global cache for frequently used objects
_PARMED_CACHE = {}
_MOLECULE_CACHE = {}

class GeomeTRICResult:
    """Container for optimization results"""
    __slots__ = ['success', 'error_code', 'message', 'final_energy', 'optimized_coords', 
                 'optimized_structure', 'n_iterations', 'convergence_info']
    
    def __init__(self, 
                 success: bool = False, 
                 error_code: int = 0, 
                 message: str = "", 
                 final_energy: float = None,
                 optimized_coords: np.ndarray = None,
                 optimized_structure: str = "",
                 n_iterations: int = 0,
                 convergence_info: Dict = None):
        self.success = success
        self.error_code = error_code  # 0=success, 1=not_converged, 2=structure_error, 3=engine_error, 4=unknown_error
        self.message = message
        self.final_energy = final_energy  # Final energy in Hartree
        self.optimized_coords = optimized_coords  # Numpy array of final coordinates in Angstrom
        self.optimized_structure = optimized_structure  # XYZ format string of final structure
        self.n_iterations = n_iterations
        self.convergence_info = convergence_info or {}


class OptimizedPySanderFromString(PySander):
    """
    Optimized PySander engine that loads AMBER files from string content.
    Eliminates file I/O and includes performance optimizations.
    """
    
    def __init__(self, molecule, prmtop_content, inpcrd_content):
        # Require a valid molecule (from parent class)
        if molecule is None:
            raise RuntimeError("OptimizedPySanderFromString engine requires a valid Molecule object")
        
        # Cache key for reusing parsed structures
        cache_key = hash(prmtop_content + inpcrd_content)
        
        # Call parent class initialization but bypass file path requirements
        self.molecule = molecule
        
        # Cache imports once per class (inherited from PySander)
        self._cache_imports()
        
        # Store string content instead of file paths
        self.prmtop_content = prmtop_content
        self.inpcrd_content = inpcrd_content
        self.prmtop_file = None  # Set to None since we're using strings
        self.inpcrd_file = None  # Set to None since we're using strings
        
        # Initialize all parent class data structures
        self.parm = None
        self.inp = None
        self.box = None
        self._coords_cache = None
        self._coords_ang_buffer = None
        self._sander_context = None
        self._context_initialized = False
        
        # Initialize Engine base class attributes
        self.stored_calcs = {}  # For caching calculations
        
        # Load from string content with caching
        self._load_parm_from_strings_cached(cache_key)
        
        # Set input (inherited method)
        self._set_input()
    
    def _load_parm_from_strings_cached(self, cache_key):
        """Load AMBER parameter and coordinate data from string content with caching"""
        
        # Check cache first
        if cache_key in _PARMED_CACHE:
            self.parm = _PARMED_CACHE[cache_key]
            n_atoms = len(self.parm.atoms)
            self._coords_ang_buffer = np.zeros((n_atoms, 3), dtype=np.float64)
            print(f" OptimizedPySanderFromString: Using cached AMBER structure with {n_atoms} atoms")
            return
        
        try:
            import parmed
            
            # Load the parmed structure from string content - TRUE zero I/O!
            self.parm = parmed.load_from_string(self.prmtop_content, self.inpcrd_content)
            
            # Cache for reuse
            _PARMED_CACHE[cache_key] = self.parm
            
            # Pre-allocate coordinate buffer for frequent conversions
            n_atoms = len(self.parm.atoms)
            self._coords_ang_buffer = np.zeros((n_atoms, 3), dtype=np.float64)
            
            #print(f" OptimizedPySanderFromString: Loaded AMBER structure with {n_atoms} atoms")
            
        except ImportError as e:
            raise RuntimeError(f"OptimizedPySanderFromString requires 'parmed' package: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load AMBER data from strings: {e}")


def create_molecule_cached(elements, initial_coords):
    """Create molecule with caching for identical topologies"""
    # Create cache key based on elements only (topology doesn't change)
    cache_key = tuple(elements)
    
    if cache_key in _MOLECULE_CACHE:
        M = _MOLECULE_CACHE[cache_key]
        # Update coordinates but reuse topology
        M.xyzs = [initial_coords.copy()]
        return M
    
    # Create new molecule
    M = Molecule()
    M.elem = elements
    M.xyzs = [initial_coords.copy()]
    M.build_topology()
    
    # Cache the molecule (without coordinates)
    M_cached = Molecule()
    M_cached.elem = elements
    M_cached.Data = M.Data.copy() if hasattr(M, 'Data') else {}
    M_cached.bonds = M.bonds.copy() if hasattr(M, 'bonds') else []
    M_cached.top = M.top if hasattr(M, 'top') else None
    
    _MOLECULE_CACHE[cache_key] = M_cached
    
    return M


def optimize_pysander_with_constraints(initial_coords: np.ndarray,
                                     bohr_coords: np.array,
                                     M: Molecule,
                                     elements: list,
                                     prmtop_content: str,
                                     inpcrd_content: str,
                                     constraints_string: str,
                                     **optimization_kwargs) -> GeomeTRICResult:
    """
    Optimize molecular geometry using PySander with constraints and zero file I/O.
    Optimized version with caching and reduced overhead.
    
    Parameters:
    -----------
    initial_coords : np.ndarray
        Initial coordinates in Angstrom, shape (n_atoms, 3)
    elements : list
        List of element symbols
    prmtop_content : str
        Content of the AMBER topology file (.prmtop) as a string
    inpcrd_content : str
        Content of the AMBER coordinate file (.inpcrd) as a string
    constraints_string : str
        Constraint specification string (same format as constraints file)
    **optimization_kwargs : dict
        Additional optimization parameters
    
    Returns:
    --------
    GeomeTRICResult : Container with optimization results
    """
    
    # Set default parameters - use more efficient defaults
    params = {
        'maxiter': 300,
        'convergence_energy': 1e-6,
        'convergence_grms': 3e-4,
        'convergence_gmax': 4.5e-4,
        'convergence_drms': 1.2e-3,
        'convergence_dmax': 1.8e-3,
        'trust_radius': 0.1,
        'coordsys': 'dlc',
        'verbose': False
    }
    params.update(optimization_kwargs)
    
    # Create temporary directory for optimization (geomeTRIC still needs a working directory)
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create molecule with caching
            #M = create_molecule_cached(elements, initial_coords)
            
            # Create optimized PySander engine 
            engine = OptimizedPySanderFromString(M, prmtop_content, inpcrd_content)
            
            # Parse constraints from string (cache could be added here too for repeated runs)
            Cons, CVals = parse_constraints(M, constraints_string)
            
            # Setup optimization parameters with optimized settings
            opt_params = OptParams(
                maxiter=params['maxiter'],
                convergence_energy=params['convergence_energy'],
                convergence_grms=params['convergence_grms'],
                convergence_gmax=params['convergence_gmax'],
                convergence_drms=params['convergence_drms'],
                convergence_dmax=params['convergence_dmax'],
                trust=params['trust_radius']
            )
            
            # Disable all file output completely
            opt_params.xyzout = None
            opt_params.qdata = None
            
            # Setup internal coordinates with constraints - optimized build
            IC = DelocalizedInternalCoordinates(M, build=True, connect=True, addcart=True, 
                                              constraints=Cons, cvals=CVals[0])
            
            # Convert coordinates to Bohr (pre-allocate for efficiency)
            #coords = np.empty(initial_coords.size, dtype=np.float64)
            #coords[:] = initial_coords.flatten() * ang2bohr
            
            # Optimized output suppression
            if not params['verbose']:
                # Use a more efficient context manager
                @contextlib.contextmanager
                def minimal_suppress():
                    old_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    try:
                        yield
                    finally:
                        sys.stdout.close()
                        sys.stdout = old_stdout
                
                # Run optimization with minimal overhead
                with minimal_suppress():
                    optimizer = Optimizer(bohr_coords, M, IC, engine, temp_dir, opt_params, print_info=False)
                    progress = optimizer.optimizeGeometry()
            else:
                optimizer = Optimizer(bohr_coords, M, IC, engine, temp_dir, opt_params, print_info=True)
                progress = optimizer.optimizeGeometry()
            
            # Extract results efficiently
            final_coords = progress.xyzs[-1]
            final_energy = progress.qm_energies[-1]
            n_iterations = len(progress.xyzs) - 1
            
            # Create XYZ format string more efficiently
            n_atoms = len(elements)
            xyz_lines = [
                str(n_atoms),
                f"Constrained optimization, Energy: {final_energy:.8f} Hartree"
            ]
            xyz_lines.extend(f"{elem:2s} {coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f}"
                           for elem, coord in zip(elements, final_coords))
            optimized_structure = '\n'.join(xyz_lines)
            
            # Check convergence
            converged = (optimizer.state == OPT_STATE.CONVERGED)
            
            # Prepare convergence info efficiently
            convergence_info = {
                'converged': converged,
                'final_state': str(optimizer.state),
                'total_iterations': n_iterations,
                'final_energy_change': abs(progress.qm_energies[-1] - progress.qm_energies[-2]) if len(progress.qm_energies) > 1 else 0.0,
                'has_constraints': bool(Cons),
                'n_constraints': len(Cons) if Cons else 0
            }
            
            if converged:
                return GeomeTRICResult(
                    success=True,
                    error_code=0,
                    message=f"Constrained optimization converged in {n_iterations} iterations",
                    final_energy=final_energy,
                    optimized_coords=final_coords,
                    optimized_structure=optimized_structure,
                    n_iterations=n_iterations,
                    convergence_info=convergence_info
                )
            else:
                return GeomeTRICResult(
                    success=False,
                    error_code=1,
                    message=f"Constrained optimization did not converge after {n_iterations} iterations",
                    final_energy=final_energy,
                    optimized_coords=final_coords,
                    optimized_structure=optimized_structure,
                    n_iterations=n_iterations,
                    convergence_info=convergence_info
                )
                
        except Exception as e:
            error_type = type(e).__name__
            if "EngineError" in error_type:
                error_code = 3
            elif "GeomOptStructureError" in error_type:
                error_code = 2
            else:
                error_code = 4
                
            return GeomeTRICResult(
                success=False,
                error_code=error_code,
                message=f"Constrained optimization failed: {error_type}: {str(e)}",
                final_energy=None,
                optimized_coords=None,
                optimized_structure="",
                n_iterations=0,
                convergence_info={}
            )

def convert_coords_to_bohr(coords):
    """
    Convert a list of coordinate arrays from Angstrom to Bohr.
    
    Parameters:
    -----------
    coords_list : list of np.ndarray
        List of coordinate arrays in Angstrom
        
    Returns:
    --------
    list of np.ndarray
        List of coordinate arrays in Bohr
    """
    import numpy as np
    
    # Conversion factor: 1 Angstrom = 1.8897259885789 Bohr
    ANGSTROM_TO_BOHR = 1.8897259885789
    
    return coords.flatten() * ANGSTROM_TO_BOHR


def create_molecule(elements, initial_coords):
    """Create molecule with caching for identical topologies"""

    # Create new molecule
    M = Molecule()
    M.elem = elements
    M.xyzs = [initial_coords.copy()]
    M.build_topology()
   
    return M


def clear_caches():
    """Clear all internal caches to free memory"""
    global _PARMED_CACHE, _MOLECULE_CACHE
    _PARMED_CACHE.clear()
    _MOLECULE_CACHE.clear()
    print("🧹 Cleared all optimization caches")


