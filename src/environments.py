import yaml

import shapely.geometry as geom
from shapely import affinity
from shapely.geometry import MultiPoint, Point, Polygon, MultiPolygon, LineString
from shapely.ops import cascaded_union, unary_union
import itertools
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from descartes import PolygonPatch
import numpy as np
import scipy.spatial

from IPython.display import SVG


color_sequence = ['r','b','y','m','c']

def color_iterator():
    i = 0
    while True:
        item = color_sequence[i]
        i += 1
        if i >= len(color_sequence): i = 0
        yield item

def save_to_yaml(data, yaml_file):
    yaml_dict = data
    f = open(yaml_file, 'w')
    f.write(yaml.dump(yaml_dict, default_flow_style=None))
    f.close()

def compute_hull(vertices):
    # Vertices are arrays N x 2
    # Create Generator matrix
    # format: each row a vertex  with a 1 appended in the first column
    mat = cdd.Matrix([[1] + list(vv)  for vv in vertices])
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    ineq_mat = poly.get_inequalities()
    # Each row in matrix is:
    # bi -ai1 -ai2 ...
    # where it represents ai1 x1 + ai2 * x2 ... <= bi
    coeffs = []
    rhss = []
    for i in range(ineq_mat.row_size):
        rhss.append(ineq_mat[i][0])
        coeffs.append([-c for c in ineq_mat[i][1:]])
    A = np.array(coeffs)
    B = np.array([rhss]).T
    if np.any(np.abs(A) > 1e2) or np.any(np.abs(A) > 1e2):
        print("WARNING: coefficients of A or B greater than 100.. probably something wrong.")
    return A, B


class Region(object):

    @classmethod
    def from_dict(cls, data):
        A = np.array(data['A'])
        B = np.array([data['B']]).T
        vertices = np.array(data['points'])
        new_reg = cls(A=A, B=B, vertices=vertices, region_id=data['region_id'])
        return new_reg

    @classmethod
    def from_vertices(cls, vertices, region_id):
        A, B = compute_hull(vertices)
        return cls(A, B, vertices, region_id)

    def __init__(self, A, B, vertices, region_id):
        # Vertices are arrays N x 2
        self.A = A
        self.B = B
        self.vertices = vertices
        #self.iris_region = iris_region
        self.region_id = region_id
        self.poly = MultiPoint(vertices).convex_hull
    def intersects(self, other_region):
        return self.poly.intersects(other_region.poly)
    def contains_point(self, point):
        return self.poly.intersects(Point(point))
    def draw(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        hull = scipy.spatial.ConvexHull(self.vertices)
        kwargs.setdefault("facecolor", "none")
        return [ax.add_patch(plt.Polygon(xy=self.vertices[hull.vertices],**kwargs))]
    def __repr__(self):
        return "<Region {}>".format(self.region_id)
    # def __eq__(self, other):
    #     if isinstance(other, self.__class__):
    #         return check_equal_vertices(self.vertices, other.vertices, tol=1e-1)
    #     return NotImplemented
    # def __ne__(self, other):
    #     """Define a non-equality test"""
    #     if isinstance(other, self.__class__):
    #         return not self.__eq__(other)
    #     return NotImplemented

    def to_dict(self):
        ir = {}
        ir['points'] = self.vertices.tolist()  # (xi,yi) each line
        ir['A'] = self.A.tolist()  # row by row
        ir['B'] = self.B[:, 0].tolist()  # transpose of B
        return ir


class SafeRegion(Region):
    def __init__(self, A, B, vertices, region_id):
        super(SafeRegion, self).__init__(A, B, vertices, region_id)
    def draw(self, ax=None, **kwargs):
        kwargs.setdefault("facecolor", "none")
        return super(SafeRegion, self).draw(ax=ax, **kwargs)
    
    def __repr__(self):
        return "<SafeRegion {}>".format(self.region_id)


def create_safe_region_from_vertices(vertices, region_id):
    A, B = compute_hull(vertices)
    return SafeRegion(A, B, vertices, region_id)


class InterestRegion(Region):

    @classmethod
    def from_dict(cls, data):
        new_reg = super(InterestRegion, cls).from_dict(data)
        if 'intersecting_safe_regions' in data:
            new_reg.intersecting_safe_regions = data['intersecting_safe_regions']
        return new_reg
    
    def __init__(self, vertices, region_id, A=None, B=None):
        # Compute A and B
        if A is None or B is None:
            A, B = compute_hull(vertices)
        super(InterestRegion, self).__init__(A, B, vertices, region_id)
        self.intersecting_safe_regions = []
        
    def draw(self, ax=None, **kwargs):
        kwargs.setdefault("facecolor", "green")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("alpha", 0.5)
        result = super(InterestRegion, self).draw(ax=ax, **kwargs)
        # Print region name
        if not ax:
            ax = plt.gca()
        x, y = np.array(self.poly.centroid.coords)[0]
        font = FontProperties()
        font.set_weight('bold')
        if len(self.region_id) <= 2:
            reg_id = self.region_id
        else:
            reg_id = self.region_id[-1]
        
        
        ax.text(x, y, "{}".format(reg_id), color='white',
                fontsize=13, fontproperties=font,
                horizontalalignment='center', verticalalignment='center')
                
        return result
    def __repr__(self):
        return "<InterestRegion {}>".format(self.region_id)
    
    
class Environment:
    def __init__(self, yaml_file=None, bounds=None):
        self.yaml_file = yaml_file
        self.environment_loaded = False

        # Iris (safe-regions)
        self.iris_regions = []
        self.iris_regions_map = dict()
        self.iris_groups = dict() # GroupId -> Set(Iris regions)

        # Interest regions
        self.interest_regions = []
        self.interest_regions_map = dict()

        # Obstacles
        # TODO: Provide groups for obstacles as well.
        self.obstacles = []
        self.obstacles_map = {}
        self.moving_obstacles = []
        self.bounds = bounds
        if not yaml_file is None:
            if self.load_from_yaml_file(yaml_file):
                if bounds is None:
                    self.calculate_scene_dimensions()
                self.environment_loaded = True
        self.obstacles_info = None
        self.obstacles_rtree_index = None

    # @property
    # def bounds(self):
    #     return self._bounds

    def get_bounds_region(self):
        left, bottom, right, top = self.bounds
        vertices = np.array([(left, bottom), (left, top), (right, top), (right, bottom)])
        return Region.from_vertices(vertices, "boundsRegion")

    def get_bounds_A_B(self):
        left, bottom, right, top = self.bounds
        return np.array([[1, 0], [0,1], [-1, 0], [0, -1]]),\
               np.array([right, top, -left, -bottom])

    def add_obstacles(self, obstacles):
        self.obstacles = self.obstacles + obstacles
        if not self.bounds:
            self.calculate_scene_dimensions()

    def compute_obstacles_info(self):
        import rtree  # Requires rtree, and libspatialindex (pip install rtree; brew install spatialindex)
        # Compute A and B for each obstacles
        # Create RTree index
        obstacles_info = dict()
        rtree_index = rtree.index.Index()
        for i,obs in enumerate(self.obstacles):
            # Compute A and B
            A, B = compute_hull(list(obs.boundary.coords))
            obstacles_info.setdefault(i, dict())['A'] = A
            obstacles_info.setdefault(i, dict())['B'] = B
            # Add to RTree index (http://toblerity.org/rtree/tutorial.html)
            # Adds the bounding box of the obstacle (left, bottom, right, top)
            rtree_index.insert(i, obs.bounds)

        self.obstacles_info = obstacles_info
        self.obstacles_rtree_index = rtree_index

    def add_moving_obstacles(self, moving_obstacles):
        self.moving_obstacles = self.moving_obstacles + moving_obstacles
    
    def add_iris_regions(self, iris_regions, group=None):
        # Clear regions first
        self.iris_regions = []
        self.iris_regions_map = {}
        for r in iris_regions:
            if r.region_id not in self.iris_regions_map:
                self.iris_regions.append(r)
                self.iris_regions_map[r.region_id] = r
                if group:
                    # Add to iris group
                    self.iris_groups.setdefault(group, set()).add(r)
            else:
                print("WARNING: {} was already in the iris regions. Not added.".
                      format(r.region_id))

    # TODO: Fix for iris groups
    def compute_iris_coverage(self):
        """Computes what portion of the obstacle-free scenario is covered by iris regions"""
        minx, miny, maxx, maxy = self.bounds
        # Generate initial volume    
        world_p = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx,miny)])
    
        # Substract obstacles
        initial_world_w_obs = world_p.difference(unary_union(self.obstacles))
        world_area = initial_world_w_obs.area
    
        working_region = initial_world_w_obs.difference(unary_union([r.poly for r in self.iris_regions]))
        coverage = 1 - working_region.area / world_area

        return coverage
    

                
    def add_interest_regions(self, interest_regions):
        for r in interest_regions:
            if r.region_id not in self.interest_regions_map:
                self.interest_regions.append(r)
                self.interest_regions_map[r.region_id] = r
            else:
                print("WARNING: {} was already in the interest regions. Not added.".
                      format(r.region_id))

    def compute_interest_region_connections(self):
        # This ignores groups
        for ir in self.interest_regions:
            intersections = []
            for r in self.iris_regions:
                if ir.intersects(r):
                    intersections.append(r)
            ir.intersecting_safe_regions = intersections

    def calculate_scene_dimensions(self):
        """Compute scene bounds from obstacles."""
        points = []
        for elem in self.obstacles:
            points = points + list(elem.boundary.coords)

        mp = geom.MultiPoint(points)
        self.bounds = mp.bounds

    def load_from_yaml_file(self, yaml_file):
        f = open(yaml_file)
        self.data = yaml.safe_load(f)
        f.close()
        return self.parse_yaml_data(self.data)

    @classmethod
    def from_dict(cls, data):
        env = cls()
        env.parse_yaml_data(data)
        return env

    def parse_yaml_data(self, data):
        if 'environment' in data:
            env = data['environment']
            self.parse_yaml_obstacles(env['obstacles'])
            if 'iris_regions' in env:
                self.parse_iris_regions(env['iris_regions'])
            if 'iris_groups' in env:
                self.parse_iris_groups(env['iris_groups'])
            if 'interest_regions' in env:
                self.parse_interest_regions(env['interest_regions'])
            if 'bounds' in env:
                self.bounds = tuple(env['bounds'])
            else:
                self.calculate_scene_dimensions()
            # self.parse_yaml_features(env['features'])
            return True
        else:
            return False

    def parse_yaml_obstacles(self, obstacles):
        self.obstacles = []
        self.obstacles_map = {}
        for ob in obstacles:
            p = Polygon([(x,y) for x,y in ob])            
            self.obstacles.append(p)

    def parse_iris_regions(self, description):
        self.iris_regions = []
        self.iris_regions_map = {}
        for r_id, r in list(description.items()):
            A = np.array(r['A'])
            B = np.array([r['B']]).T
            vertices = np.array(r['points'])            
            my_reg = SafeRegion(A, B, vertices, r_id)
            self.iris_regions.append(my_reg)
            self.iris_regions_map[r_id] = my_reg

    def parse_iris_groups(self, description):
        iris_groups = dict()
        for group, regions in description.items():
            iris_groups[group] = [self.iris_regions_map[rid] for rid in regions]
        self.iris_groups = iris_groups


    def parse_interest_regions(self, description):
        self.interest_regions = []
        self.interest_regions_map = {}
        for r_id, r in list(description.items()):
            # Duplicate to avoid making changes to original
            rd = dict(r)
            rd['region_id'] = r_id
            my_reg = InterestRegion.from_dict(rd)
            self.interest_regions.append(my_reg)
            self.interest_regions_map[r_id] = my_reg

    def to_dict(self, planner=None):
        env_dict = {}
        obstacles = []
        for i, ob in enumerate(self.obstacles):
            vert_list = []  # [(xi,yi)]
            for xi, yi in zip(*np.array(ob.exterior.coords).T.tolist()):
                vert_list.append([xi, yi])
            obstacles.append(vert_list)
            
        interest_dict = {}
        for i, r in enumerate(self.interest_regions):
            interest_dict[r.region_id] = r.to_dict()
            if planner == 'psulu':
                del interest_dict[r.region_id]['points']
        
        if planner=="scottypath"or planner == None:
            iris_dict = {}
            for i, r in enumerate(self.iris_regions):
                ir = {}
                # ip = r.iris_region.getPolyhedron()
                ir['points'] = r.vertices.tolist()  # (xi,yi) each line
                ir['A'] = r.A.tolist()  # row by row
                ir['B'] = r.B[:, 0].tolist()  # transpose of B
                iris_dict[r.region_id] = ir

            # Iris groups
            iris_groups = dict()
            for group, regions in self.iris_groups.items():
                iris_groups[group] = [r.region_id for r in regions]



            env_dict['environment'] = {'obstacles': obstacles,
                                        'iris_regions': iris_dict,
                                        'iris_groups': iris_groups,
                                        'interest_regions': interest_dict,
                                        'bounds': list(self.bounds)
                                        }
        elif planner=="psulu":
            env_dict['environment'] = {'obstacles': obstacles,
                                        'interest_regions': interest_dict}
        
        return env_dict

    def save_to_yaml(self, yaml_file):
        yaml_dict = self.to_dict()
        f = open(yaml_file, 'w')
        f.write(yaml.dump(yaml_dict, default_flow_style=None))
        f.close()
        
    def make_figure_axes(self, bounds=None, figsize = None):

        if bounds:
            minx, miny, maxx, maxy = bounds
        else:
            minx, miny, maxx, maxy = self.bounds

        max_width, max_height = 12, 5.5
        if figsize is None:
            width, height = max_width, (maxy-miny)*max_width/(maxx-minx)
            if height > 5:
                width, height = (maxx-minx)*max_height/(maxy-miny), max_height
            figsize = (width, height)

        f = plt.figure(figsize=figsize)
        # f.hold('on')
        ax = f.add_subplot(111)

        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        bounds = minx, miny, maxx, maxy
        return f, ax, bounds, figsize

    def draw(self, bounds=None, figsize=None, draw_interest_regions=True, draw_iris=False, figure=None, aspect_equal=True, **kwargs):
        # if bounds:
        #     minx, miny, maxx, maxy = bounds
        # else:
        #     minx, miny, maxx, maxy = self.bounds
        #
        # max_width, max_height = 12, 5.5
        # if figsize is None:
        #     width, height = max_width, (maxy-miny)*max_width/(maxx-minx)
        #     if height > 5:
        #         width, height = (maxx-minx)*max_height/(maxy-miny), max_height
        #     figsize = (width, height)

        kwargs.setdefault('obstacles_color', 'blue')

        if not figure:
            f, ax, bounds, figsize = self.make_figure_axes(bounds=bounds, figsize=figsize)
            minx, miny, maxx, maxy = bounds
        else:
            f = figure
            # f.hold('on')
            ax = figure.axes[0]
            minx, maxx = ax.get_xlim()
            miny, maxy = ax.get_ylim()
            # print("Max/min: ", minx, maxx, miny, maxy)

        for i, obs in enumerate(self.obstacles):
            patch = PolygonPatch(obs, fc=kwargs['obstacles_color'], ec=kwargs['obstacles_color'], alpha=0.5, zorder=20)
            ax.add_patch(patch)

        if draw_interest_regions:
            if isinstance(draw_interest_regions, list):
                for r in draw_interest_regions: self.interest_regions_map[r].draw()
            else:
                for r in self.interest_regions: r.draw()
        if draw_iris:
            if draw_iris is True:
                for r in self.iris_regions: r.draw(edgecolor='r', linestyle='-',  facecolor='r', alpha=0.05)
            elif isinstance(draw_iris, str) or isinstance(draw_iris, list):
                if isinstance(draw_iris, str):
                    draw_iris = [draw_iris]
                color_it = color_iterator()
                for group, col in zip(draw_iris, color_it):
                    for r in self.iris_groups[group]:
                        r.draw(edgecolor=col, linestyle='-', facecolor=col, alpha=0.05)
                # for r in self.iris_regions: r.draw(edgecolor='k',linestyle='--')

        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        if aspect_equal:
            ax.set_aspect('equal', adjustable='box')
        return ax

    def draw_time(self, time, bounds=None, figsize=None, draw_interest_regions=True, draw_iris=False, figure=None):
        ax = self.draw(bounds, figsize, draw_interest_regions, draw_iris, figure=figure)
        for mo in self.moving_obstacles:
            mo.draw(time, ax=ax)
    
