"""Utilities"""

def pcie(spec: str) -> float:
    """Parse PCIe speficiation, returning the bandwidth in GB/s"""

    spec = spec.lower().strip()
    if 'x' in spec:
        gen, lanes = spec.split('x')
        try:
            lanes = int(lanes)
            if lanes not in [1, 2, 4, 8, 16]:
                raise Exception()
            match gen:
                case 'gen1' | '1': return 0.25 * lanes
                case 'gen2' | '2': return 0.5 * lanes
                case 'gen3' | '3': return 1.0 * lanes
                case 'gen4' | '4': return 2.0 * lanes
                case _: raise Exception()
        except:
            pass
    else:
        try:
            bw = float(spec)
            return bw
        except:
            pass
        
    raise ValueError(f"unrecognized PCIe specification: {spec}")
    
    