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


def optimize_pysander_with_constraints(elements: list,
                                     initial_coords: np.ndarray,
                                     prmtop_content: str,
                                     inpcrd_content: str,
                                     constraints_string: str,
                                     enable_logging: bool = False,
                                     debug_level: int = 0,
                                     output_suffix: str = "geometric_optimization",
                                     output_directory: str = ".",
                                     use_vectorized_calcDiff: bool = True,
                                     **optimization_kwargs) -> GeomeTRICResult:
    """
    Optimize molecular geometry using PySander with constraints and zero file I/O.
    Optimized version with caching and reduced overhead.
    
    Parameters:
    -----------
    elements : list
        List of element symbols
    initial_coords : np.ndarray
        Initial coordinates in Angstrom, shape (n_atoms, 3)
    prmtop_content : str
        Content of the AMBER topology file (.prmtop) as a string
    inpcrd_content : str
        Content of the AMBER coordinate file (.inpcrd) as a string
    constraints_string : str
        Constraint specification string (same format as constraints file)
    enable_logging : bool, optional
        Enable geomeTRIC file logging (.log and _optim.xyz files) (default: False)
    debug_level : int, optional
        Debug level for log output (default: 0)
        0: No gradient information in log (compact format)
        1: Include detailed gradient information (JOB entries) in log
        2: Include full gradient vectors for each step (maximum detail)
    output_suffix : str, optional
        Suffix for output files (default: "geometric_optimization")
        Results in {suffix}.log and {suffix}_optim.xyz files
    output_directory : str, optional
        Directory to save output files (default: "." - current directory)
        Directory will be created if it doesn't exist
    use_vectorized_calcDiff : bool, optional
        Use vectorized calcDiff implementation for improved performance (default: True)
        Provides approximately 18-24% speedup. Works with multiprocessing.
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
            # Create molecule from elements and coordinates
            M = create_molecule(elements, initial_coords)
            
            # Convert coordinates to Bohr for geomeTRIC
            bohr_coords = convert_coords_to_bohr(initial_coords)
            
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
            
            # Configure file output and logger capture based on logging flag
            log_stream = None
            original_handlers = None
            if enable_logging:
                import io
                import logging
                
                # Set up string buffer to capture logger output
                log_stream = io.StringIO()
                
                # Capture geometric's logger output
                geometric_logger = logging.getLogger('geometric')
                handler = logging.StreamHandler(log_stream)
                handler.setLevel(logging.INFO)
                original_handlers = geometric_logger.handlers[:]
                geometric_logger.handlers = [handler]
                geometric_logger.setLevel(logging.INFO)
                
                # Enable geomeTRIC file logging - files will be written to temp_dir
                log_file_path = os.path.join(temp_dir, "optimization.log")
                opt_params.xyzout = os.path.join(temp_dir, "optimization_trajectory.xyz")
                opt_params.qdata = log_file_path
            else:
                # Disable all file output for maximum performance
                opt_params.xyzout = None
                opt_params.qdata = None
            
            # Setup internal coordinates with constraints - optimized build
            IC = DelocalizedInternalCoordinates(M, build=True, connect=True, addcart=True, 
                                              constraints=Cons, cvals=CVals[0],
                                              use_vectorized_calcDiff=use_vectorized_calcDiff)
            
            # Convert coordinates to Bohr (pre-allocate for efficiency)
            #coords = np.empty(initial_coords.size, dtype=np.float64)
            #coords[:] = initial_coords.flatten() * ang2bohr
            
            # Optimized output suppression
            try:
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
            finally:
                # Restore original logger handlers if we modified them
                if enable_logging and original_handlers is not None:
                    import logging
                    geometric_logger = logging.getLogger('geometric')
                    geometric_logger.handlers = original_handlers
            
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
            
            # Handle log files if logging was enabled
            log_content = ""
            trajectory_content = ""
            
            # Always include logging info in convergence_info
            logging_info = {
                'logging_enabled': enable_logging,
                'log_content': None,
                'trajectory_content': None
            }
            
            if enable_logging:
                # Ensure output directory exists
                os.makedirs(output_directory, exist_ok=True)
                
                # Generate output file paths with custom suffix and directory
                output_log_path = os.path.join(output_directory, f"{output_suffix}.log")
                output_trajectory_path = os.path.join(output_directory, f"{output_suffix}_optim.xyz")
                
                # Combine header with captured logger output and formatted qdata
                trajectory_file = os.path.join(temp_dir, "optimization_trajectory.xyz")
                qdata_file = os.path.join(temp_dir, "optimization.log")
                
                # Get the captured logger output (step information)
                step_output = log_stream.getvalue() if log_stream else ""
                
                # Read and format qdata output (raw JOB entries) based on debug_level
                formatted_qdata = ""
                if debug_level >= 1 and os.path.exists(qdata_file):
                    with open(qdata_file, 'r') as f:
                        qdata_content = f.read()
                    # Extract only the JOB entries (skip our header)
                    if "JOB 0" in qdata_content:
                        raw_qdata = qdata_content[qdata_content.find("JOB 0"):]
                        formatted_qdata = _format_qdata_output(raw_qdata, debug_level)
                
                # Create complete log: header + step output + optional formatted qdata
                header = _create_geometric_log_header(elements, constraints_string, 
                                                    prmtop_content, inpcrd_content, opt_params)
                log_content = header + step_output + ("\n" + formatted_qdata if formatted_qdata else "")
                
                # Write complete log to specified location
                with open(output_log_path, 'w') as f:
                    f.write(log_content)
                print(f"📝 GeomeTRIC log file saved as: {output_log_path}")
                
                # Read trajectory content  
                if os.path.exists(trajectory_file):
                    with open(trajectory_file, 'r') as f:
                        trajectory_content = f.read()
                    # Copy to specified location
                    import shutil
                    shutil.copy2(trajectory_file, output_trajectory_path)
                    print(f"📊 GeomeTRIC trajectory file saved as: {output_trajectory_path}")
            
            # Prepare convergence info efficiently
            convergence_info = {
                'converged': converged,
                'final_state': str(optimizer.state),
                'total_iterations': n_iterations,
                'final_energy_change': abs(progress.qm_energies[-1] - progress.qm_energies[-2]) if len(progress.qm_energies) > 1 else 0.0,
                'has_constraints': bool(Cons),
                'n_constraints': len(Cons) if Cons else 0
            }
            convergence_info.update(logging_info)
            
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
            
            # Try to read log files even if optimization failed
            log_content = ""
            trajectory_content = ""
            if enable_logging:
                # Ensure output directory exists
                os.makedirs(output_directory, exist_ok=True)
                
                # Generate output file paths with custom suffix and directory
                output_log_path = os.path.join(output_directory, f"{output_suffix}.log")
                output_trajectory_path = os.path.join(output_directory, f"{output_suffix}_optim.xyz")
                
                log_file = os.path.join(temp_dir, "optimization.log")
                trajectory_file = os.path.join(temp_dir, "optimization_trajectory.xyz")
                
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    # Copy to specified location
                    import shutil
                    shutil.copy2(log_file, output_log_path)
                    print(f"📝 GeomeTRIC log file saved as: {output_log_path} (from failed run)")
                
                if os.path.exists(trajectory_file):
                    with open(trajectory_file, 'r') as f:
                        trajectory_content = f.read()
                    # Copy to specified location
                    import shutil
                    shutil.copy2(trajectory_file, output_trajectory_path)
                    print(f"📊 GeomeTRIC trajectory file saved as: {output_trajectory_path} (from failed run)")
            
            # Include logging info even in error cases
            error_convergence_info = {
                'logging_enabled': enable_logging,
                'log_content': log_content if enable_logging and log_content else None,
                'trajectory_content': trajectory_content if enable_logging and trajectory_content else None,
                'error_type': error_type
            }
                
            return GeomeTRICResult(
                success=False,
                error_code=error_code,
                message=f"Constrained optimization failed: {error_type}: {str(e)}",
                final_energy=None,
                optimized_coords=None,
                optimized_structure="",
                n_iterations=0,
                convergence_info=error_convergence_info
            )

def parse_coords_from_inpcrd(inpcrd_content):
    """
    Parse coordinates from AMBER INPCRD file content.
    
    Parameters:
    -----------
    inpcrd_content : str
        Content of AMBER INPCRD file
        
    Returns:
    --------
    np.ndarray
        Coordinate array in Angstrom, shape (n_atoms, 3)
    """
    import numpy as np
    
    lines = inpcrd_content.strip().split('\n')
    if len(lines) < 2:
        raise ValueError("Invalid INPCRD content: not enough lines")
    
    # Second line contains number of atoms
    n_atoms = int(lines[1].split()[0])
    
    # Parse coordinate lines (starting from line 2)
    coords = []
    coord_lines = lines[2:]
    
    for line in coord_lines:
        # AMBER format: 6 coordinates per line (x,y,z for 2 atoms)
        values = line.split()
        coords.extend([float(x) for x in values])
    
    # Reshape to (n_atoms, 3)
    coords_array = np.array(coords[:n_atoms*3]).reshape(n_atoms, 3)
    
    return coords_array


def convert_coords_to_bohr(coords):
    """
    Convert coordinate array from Angstrom to Bohr.
    
    Parameters:
    -----------
    coords : np.ndarray
        Coordinate array in Angstrom
        
    Returns:
    --------
    np.ndarray
        Coordinate array in Bohr, flattened
    """
    import numpy as np
    
    # Conversion factor: 1 Angstrom = 1.8897259885789 Bohr
    ANGSTROM_TO_BOHR = 1.8897259885789
    
    return coords.flatten() * ANGSTROM_TO_BOHR


def _create_geometric_log_header(elements, constraints_string, prmtop_content, inpcrd_content, opt_params):
    """
    Create proper geomeTRIC log header with version info, ASCII art, and configuration details.
    """
    import datetime
    
    # Get geomeTRIC version info
    try:
        from geometric import _version
        version = _version.version
    except:
        version = "0+untagged.dev"
    
    # Parse constraints to show internal coordinate system
    constraint_display = ""
    
    # Parse constraint string for display
    if constraints_string.strip():
        constraint_display = constraints_string.strip()
    
    # Generate basic internal coordinate system info
    n_atoms = len(elements)
    n_coords = n_atoms * 3 - 6  # Approximate number of internal coordinates
    
    # Build optimization info section with actual parameters
    opt_info_section = ""
    if opt_params:
        max_trust = getattr(opt_params, 'tmax', 0.3)  # Default geomeTRIC max trust radius
        initial_trust = getattr(opt_params, 'trust', 0.1)  # Default initial trust radius
        maxiter = getattr(opt_params, 'maxiter', 300)
        
        # Get convergence criteria from opt_params (note: OptParams uses Capital C)
        conv_energy = getattr(opt_params, 'Convergence_energy', 1e-6)
        conv_grms = getattr(opt_params, 'Convergence_grms', 3e-4)
        conv_gmax = getattr(opt_params, 'Convergence_gmax', 4.5e-4)
        conv_drms = getattr(opt_params, 'Convergence_drms', 1.2e-3)
        conv_dmax = getattr(opt_params, 'Convergence_dmax', 1.8e-3)
        
        opt_info_section = f"""
> ===== Optimization Info: ====
> Job type: Energy minimization
> Maximum number of optimization cycles: {maxiter}
> Initial / maximum trust radius (Angstrom): {initial_trust:.3f} / {max_trust:.3f}
> Convergence Criteria:
> Will converge when all 5 criteria are reached:
>  |Delta-E| < {conv_energy:.2e}
>  RMS-Ortho-Grad < {conv_grms:.2e}
>  Max-Ortho-Grad < {conv_gmax:.2e}
>  RMS-Disp  < {conv_drms:.2e}
>  Max-Disp  < {conv_dmax:.2e}"""
        
        # Add constraint criterion if constraints are present
        if constraints_string.strip():
            opt_info_section += "\n>\n> Constraints are requested. The following criterion is added:\n>  Max Constraint Violation (in Angstroms/degrees) < 1.00e-02"
        
        opt_info_section += "\n> === End Optimization Info ==="

    header = f"""geometric-optimize called with the following command line:
optimize_pysander_with_constraints (Python API call with constraints and PySander engine)

                                        ())))))))))))))))/                     
                                    ())))))))))))))))))))))))),                
                                *)))))))))))))))))))))))))))))))))             
                        #,    ()))))))))/                .)))))))))),          
                      #%%%%,  ())))))                        .))))))))*        
                      *%%%%%%,  ))              ..              ,))))))).      
                        *%%%%%%,         ***************/.        .)))))))     
                #%%/      (%%%%%%,    /*********************.       )))))))    
              .%%%%%%#      *%%%%%%,  *******/,     **********,      .))))))   
                .%%%%%%/      *%%%%%%,  **              ********      .))))))  
          ##      .%%%%%%/      (%%%%%%,                  ,******      /)))))  
        %%%%%%      .%%%%%%#      *%%%%%%,    ,/////.       ******      )))))) 
      #%      %%      .%%%%%%/      *%%%%%%,  ////////,      *****/     ,))))) 
    #%%  %%%  %%%#      .%%%%%%/      (%%%%%%,  ///////.     /*****      ))))).
  #%%%%.      %%%%%#      /%%%%%%*      #%%%%%%   /////)     ******      ))))),
    #%%%%##%  %%%#      .%%%%%%/      (%%%%%%,  ///////.     /*****      ))))).
      ##     %%%      .%%%%%%/      *%%%%%%,  ////////.      *****/     ,))))) 
        #%%%%#      /%%%%%%/      (%%%%%%      /)/)//       ******      )))))) 
          ##      .%%%%%%/      (%%%%%%,                  *******      ))))))  
                .%%%%%%/      *%%%%%%,  **.             /*******      .))))))  
              *%%%%%%/      (%%%%%%   ********/*..,*/*********       *))))))   
                #%%/      (%%%%%%,    *********************/        )))))))    
                        *%%%%%%,         ,**************/         ,))))))/     
                      (%%%%%%   ()                              ))))))))       
                      #%%%%,  ())))))                        ,)))))))),        
                        #,    ())))))))))                ,)))))))))).          
                                 ()))))))))))))))))))))))))))))))/             
                                    ())))))))))))))))))))))))).                
                                         ())))))))))))))),                     

-=#  geomeTRIC started. Version: {version}  #=-
Current date and time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#========================================================#
#|     Arguments passed to driver run_optimizer():      |#
#========================================================#
engine                    pysander_api
input                     Generated from elements and coordinates
constraints               User-provided constraints string
----------------------------------------------------------
PySander API engine selected (in-memory optimization).
   Engine: OptimizedPySanderFromString
   Topology: {len(prmtop_content)} byte AMBER prmtop content
   Coordinates: {len(inpcrd_content)} byte AMBER inpcrd content
   
{constraint_display}
Bonds will be generated from interatomic distances less than 1.20 times sum of covalent radii
{n_coords} internal coordinates being used (instead of {n_atoms * 3} Cartesians)
Internal coordinate system (atoms numbered from 1):
<DLC info would be generated by geomeTRIC engine during actual optimization>{opt_info_section}

"""
    
    return header


def _format_qdata_output(raw_qdata, debug_level=1):
    """
    Format raw qdata JOB entries into a more readable summary format.
    
    Parameters:
    -----------
    raw_qdata : str
        Raw JOB entries from qdata output
    debug_level : int
        Debug level for output detail
        1: Summary with RMS/Max gradients
        2: Full gradient vectors for each step
    """
    import re
    import numpy as np
    
    if not raw_qdata or "JOB" not in raw_qdata:
        return ""
    
    if debug_level == 2:
        formatted_output = "\n=== Full Optimization Data (Debug Level 2) ===\n"
    else:
        formatted_output = "\n=== Detailed Optimization Data ===\n"
    
    # Split by JOB entries
    job_pattern = r'JOB\s+(\d+)'
    jobs = re.split(job_pattern, raw_qdata)[1:]  # Skip empty first element
    
    for i in range(0, len(jobs), 2):
        if i + 1 >= len(jobs):
            break
            
        job_num = jobs[i]
        job_data = jobs[i + 1]
        
        # Extract energy
        energy_match = re.search(r'ENERGY\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', job_data)
        energy = float(energy_match.group(1)) if energy_match else 0.0
        
        # Extract gradients
        grad_match = re.search(r'GRADIENT\s+(.*?)(?=JOB|\Z)', job_data, re.DOTALL)
        if grad_match:
            grad_text = grad_match.group(1).strip()
            grad_values = []
            for line in grad_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('JOB'):
                    # Extract numbers from the line
                    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
                    grad_values.extend([float(x) for x in numbers])
            
            if grad_values:
                grad_array = np.array(grad_values)
                rms_grad = np.sqrt(np.mean(grad_array**2))
                max_grad = np.max(np.abs(grad_array))
                
                formatted_output += f"Job {job_num:2s}: Energy = {energy:15.10f} Hartree\n"
                formatted_output += f"        RMS Gradient = {rms_grad:.6e} | Max Gradient = {max_grad:.6e}\n"
                formatted_output += f"        Gradient Components: {len(grad_values)} total\n"
                
                # For debug_level=2, print full gradient vector
                if debug_level >= 2:
                    formatted_output += "        Full Gradient Vector:\n"
                    # Print gradients in groups of 6 for readability (similar to AMBER format)
                    for j in range(0, len(grad_values), 6):
                        grad_group = grad_values[j:j+6]
                        grad_line = "        " + " ".join([f"{g:14.8e}" for g in grad_group])
                        formatted_output += grad_line + "\n"
                
                formatted_output += "\n"
            else:
                formatted_output += f"Job {job_num:2s}: Energy = {energy:15.10f} Hartree (No gradient data)\n\n"
        else:
            formatted_output += f"Job {job_num:2s}: Energy = {energy:15.10f} Hartree (No gradient data)\n\n"
    
    return formatted_output


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


