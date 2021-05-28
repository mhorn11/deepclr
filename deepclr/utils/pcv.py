from datetime import datetime
import os
import os.path as osp
from typing import Dict, List, Optional, Union
import warnings

import numpy as np
import vtk
import vtkmodules


class _VtkPointCloud:
    """Point cloud data structure."""
    def __init__(self, points: Optional[np.ndarray] = None):
        # Init members
        self._vtk_poly_data = vtk.vtkPolyData()
        self._vtk_points = vtk.vtkPoints()
        self._vtk_cells = vtk.vtkCellArray()
        self._vtk_depth = vtk.vtkDoubleArray()

        # Initialize cloud
        self.clear_points()

        # Create actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._vtk_poly_data)
        mapper.SetColorModeToDefault()
        mapper.ScalarVisibilityOff()
        self._vtk_actor = vtk.vtkActor()
        self._vtk_actor.SetMapper(mapper)

        # Add points
        if points is not None:
            self.add_points(points)

        self.modified()

    def get_actor(self) -> vtk.vtkActor:
        return self._vtk_actor

    def add_point(self, point: np.ndarray) -> None:
        # Add point
        point_id = self._vtk_points.InsertNextPoint(point[:])
        self._vtk_depth.InsertNextValue(point[2])
        self._vtk_cells.InsertNextCell(1)
        self._vtk_cells.InsertCellPoint(point_id)

    def add_points(self, points: np.ndarray) -> None:
        # Add points
        points = np.array(points)
        for i in range(points.shape[0]):
            self.add_point(points[i, :])

    def set_color(self, r: float, g: float, b: float) -> None:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._vtk_poly_data)
        mapper.SetColorModeToDefault()
        mapper.ScalarVisibilityOff()
        self._vtk_actor.SetMapper(mapper)
        self._vtk_actor.GetProperty().SetColor(r, g, b)

    def set_colors(self, color: np.ndarray) -> None:
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Colors')
        for k in range(color.shape[0]):
            colors.InsertNextTypedTuple(color[k, :].astype(np.uint8))
        self._vtk_poly_data.GetPointData().SetScalars(colors)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._vtk_poly_data)
        self._vtk_actor.SetMapper(mapper)
        self._vtk_actor.GetProperty().SetOpacity(1.0)

    def set_point_size(self, s: float) -> None:
        self._vtk_actor.GetProperty().SetPointSize(s)

    def modified(self) -> None:
        self._vtk_cells.Modified()
        self._vtk_points.Modified()
        self._vtk_depth.Modified()

    def clear_points(self) -> None:
        self._vtk_points = vtk.vtkPoints()
        self._vtk_cells = vtk.vtkCellArray()
        self._vtk_depth = vtk.vtkDoubleArray()
        self._vtk_depth.SetName('DepthArray')
        self._vtk_poly_data.SetPoints(self._vtk_points)
        self._vtk_poly_data.SetVerts(self._vtk_cells)
        self._vtk_poly_data.GetPointData().SetScalars(self._vtk_depth)
        self._vtk_poly_data.GetPointData().SetActiveScalars('DepthArray')


class _VtkPolygon:
    """Polygon data structure."""
    def __init__(self, points: np.ndarray):
        # Init members
        self._vtk_points = vtk.vtkPoints()
        self._vtk_polygon = vtk.vtkPolygon()

        # Create the polygon
        self._vtk_polygon.GetPointIds().SetNumberOfIds(points.shape[0])
        for i in range(points.shape[0]):
            point_id = self._vtk_points.InsertNextPoint(points[i, :])
            self._vtk_polygon.GetPointIds().SetId(point_id, point_id)

        # Add the polygon to a list of polygons
        self._vtk_cells = vtk.vtkCellArray()
        self._vtk_cells.InsertNextCell(self._vtk_polygon)

        # Create PolyData
        self._vtk_poly_data = vtk.vtkPolyData()
        self._vtk_poly_data.SetPoints(self._vtk_points)
        self._vtk_poly_data.SetPolys(self._vtk_cells)

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._vtk_poly_data)
        mapper.SetColorModeToDefault()
        mapper.ScalarVisibilityOff()

        self._vtk_actor = vtk.vtkActor()
        self._vtk_actor.SetMapper(mapper)

    def get_actor(self) -> vtk.vtkActor:
        return self._vtk_actor

    def set_color(self, r: float, g: float, b: float) -> None:
        self._vtk_actor.GetProperty().SetColor(r, g, b)

    def set_alpha(self, a: float) -> None:
        self._vtk_actor.GetProperty().SetOpacity(a)


class PointCloudVisualizer:
    """Point cloud viewer based on vtk."""
    _axes_marker: vtk.vtkOrientationMarkerWidget
    _ground_plane: Optional[_VtkPolygon]

    def __init__(self, name: str = 'PointCloudVisualizer'):
        # Point Clouds and ground plane
        self._clouds: Dict[str, _VtkPointCloud] = dict()
        self._axes_marker = None
        self._ground_plane = None

        # Renderer
        self._renderer = vtk.vtkRenderer()
        self._renderer.ResetCamera()

        # Render Window
        self._render_window = vtk.vtkRenderWindow()
        self._render_window.AddRenderer(self._renderer)
        self._render_window.setWindowName = name

        # Interactor
        self._render_window_interactor = vtk.vtkRenderWindowInteractor()
        self._render_window_interactor.SetRenderWindow(self._render_window)
        self._render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self._render_window_interactor.AddObserver('KeyPressEvent', self._key_press_event)
        self._render_window_interactor.AddObserver('ExitEvent', self._close_event)
        self._render_window_interactor.Initialize()

        # Timer Id
        self._timer_id = None

        # Axes Marker
        self._axes_marker = vtk.vtkOrientationMarkerWidget()
        self._init_axes_marker()
        self.show_axes_marker(True)

        # Default values
        self._is_closed = False
        self._set_default_camera_params()

    def set_window_size(self, x: int, y: int) -> None:
        self._render_window.SetSize(x, y)

    def set_background(self, r: float, g: float, b: float) -> None:
        self._renderer.SetBackground(r, g, b)

    def add_point_cloud(self, identifier: str, cloud: np.ndarray,
                        color: Optional[Union[List[float], np.ndarray]] = None, size: Optional[float] = None) -> bool:
        if identifier in self._clouds.keys():
            warnings.warn("Cloud '{}' already exists.".format(identifier))
            return False

        if color is None:
            color = [1.0, 1.0, 1.0]
        if size is None:
            size = 1

        vtk_cloud = _VtkPointCloud(cloud)
        self._set_color(vtk_cloud, color)
        vtk_cloud.set_point_size(size)

        self._clouds[identifier] = vtk_cloud
        self._renderer.AddActor(self._clouds[identifier].get_actor())
        return True

    def update_point_cloud(self, identifier: str, cloud: Optional[np.ndarray] = None,
                           color: Optional[Union[List[float], np.ndarray]] = None, size: Optional[float] = None) \
            -> bool:
        if identifier not in self._clouds.keys():
            if cloud is not None:
                self.add_point_cloud(identifier, cloud, color=color, size=size)
                return True
            else:
                warnings.warn("Cloud '{}' does not exist.".format(identifier))
                return False

        vtk_cloud = self._clouds[identifier]

        if cloud is not None:
            vtk_cloud.clear_points()
            vtk_cloud.add_points(cloud)

        if color is not None:
            self._set_color(vtk_cloud, color)

        if size is not None:
            vtk_cloud.set_point_size(size)

        vtk_cloud.modified()
        return True

    @staticmethod
    def _set_color(vtk_cloud: _VtkPointCloud, color: Union[List[float], np.ndarray]) -> None:
        if isinstance(color, list) or color.shape[0] == 1:
            vtk_cloud.set_color(color[0], color[1], color[2])
        else:
            vtk_cloud.set_colors(color)

    def remove_point_cloud(self, identifier: str) -> None:
        if identifier in self._clouds:
            self._renderer.RemoveActor(self._clouds[identifier].get_actor())
            del self._clouds[identifier]

    def remove_all_point_clouds(self) -> None:
        cloud_ids = [key for key in self._clouds.keys()]
        for id in cloud_ids:
            self.remove_point_cloud(id)

    def _init_axes_marker(self) -> None:
        axes = vtk.vtkAxesActor()
        self._axes_marker.SetOutlineColor(0.9300, 0.5700, 0.1300)
        self._axes_marker.SetOrientationMarker(axes)
        self._axes_marker.SetInteractor(self._render_window_interactor)
        self._axes_marker.SetViewport(0.0, 0.0, 0.3, 0.3)
        self._axes_marker.SetEnabled(0)
        self._axes_marker.InteractiveOn()

    def show_axes_marker(self, show: bool) -> None:
        if show:
            self._axes_marker.SetEnabled(1)
        else:
            self._axes_marker.SetEnabled(0)

    def set_ground_plane(self, show: bool, length: float = 5.0,
                         color: Optional[Union[List[float], np.ndarray]] = None, alpha: Optional[float] = None) -> None:
        # Change values and keep plane existing
        if show and self._ground_plane is not None:
            if color is not None:
                self._ground_plane.set_color(color[0], color[1], color[2])
            if alpha is not None:
                self._ground_plane.set_alpha(alpha)
            return

        # Create new plane
        if show and self._ground_plane is None:
            x = length
            pts = np.array([
                [x, x, 0.0],
                [-x, x, 0.0],
                [-x, -x, 0.0],
                [x, -x, 0.0],
            ])
            self._ground_plane = _VtkPolygon(pts)
            self._renderer.AddActor(self._ground_plane.get_actor())

            if color is None:
                color = np.array([1.0, 1.0, 1.0])
            if alpha is None:
                alpha = 1.0

            self._ground_plane.set_color(color[0], color[1], color[2])
            self._ground_plane.set_alpha(alpha)
            return

        # Remove plane
        if not show and self._ground_plane is not None:
            self._renderer.RemoveActor(self._ground_plane.get_actor())
            self._ground_plane = None
            return

    def get_camera_params(self) -> Dict:
        position = self._renderer.GetActiveCamera().GetPosition()
        focal_point = self._renderer.GetActiveCamera().GetFocalPoint()
        view_up = self._renderer.GetActiveCamera().GetViewUp()
        return {'position': position, 'focal_point': focal_point, 'view_up': view_up}

    def set_camera_params(self, position: Optional[List[float]] = None,
                          focal_point: Optional[List[float]] = None,
                          view_up: Optional[List[float]] = None) -> None:
        if position is not None:
            self._renderer.GetActiveCamera().SetPosition(position[0], position[1], position[2])
        if focal_point is not None:
            self._renderer.GetActiveCamera().SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
        if view_up is not None:
            self._renderer.GetActiveCamera().SetViewUp(view_up[0], view_up[1], view_up[2])

    def spin(self) -> None:
        if self._is_closed:
            raise RuntimeError("Visualizer is closed")

        self._render_window.Render()
        self._render_window_interactor.Start()

    def spin_once(self, t: float, force_redraw: bool = True) -> None:
        if self._is_closed:
            raise RuntimeError("Visualizer is closed")

        if force_redraw:
            self._render_window_interactor.Render()

        self._render_window_interactor.AddObserver('TimerEvent', self._timer_event)
        self._timer_id = self._render_window_interactor.CreateRepeatingTimer(t)

        self._render_window.Render()
        self._render_window_interactor.Start()
        self._render_window_interactor.DestroyTimer(self._timer_id)

    def close(self) -> None:
        self._axes_marker.SetEnabled(0)
        self._render_window.Finalize()
        self._render_window_interactor.TerminateApp()
        del self._render_window, self._render_window_interactor
        self._is_closed = True

    def _timer_event(self, _obj: vtkmodules.vtkRenderingUI.vtkXRenderWindowInteractor, _ev: str) -> None:
        self._render_window_interactor.TerminateApp()

    def _key_press_event(self, obj: vtkmodules.vtkRenderingUI.vtkXRenderWindowInteractor, _ev: str) -> None:
        print(type(obj))
        print(type(_ev))
        ctrl = obj.GetControlKey()
        key_sym = obj.GetKeySym()
        if ctrl and key_sym == 'c':
            print(self.get_camera_params())
        elif ctrl and key_sym == 's':
            self._save_screenshot()
        elif ctrl and key_sym == 'x':
            self._axes_marker.SetEnabled(not self._axes_marker.GetEnabled())

    def _save_screenshot(self) -> None:
        directory = osp.expanduser('~/Pictures/PointCloudVisualizer')
        name = datetime.now().strftime('screenshot_%Y%M%d%H%M%S%f.png')
        filename = osp.join(directory, name)

        render_large = vtk.vtkRenderLargeImage()
        render_large.SetInput(self._renderer)
        render_large.SetMagnification(5)

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(render_large.GetOutputPort())

        os.makedirs(directory, exist_ok=True)
        writer.Write()
        print("Screenshot saved to '{}'".format(filename))

    def _close_event(self, _obj: vtkmodules.vtkRenderingUI.vtkXRenderWindowInteractor, _ev: str) -> None:
        self.close()

    def _set_default_camera_params(self) -> None:
        # Camera
        camera = vtk.vtkCamera()
        camera.SetPosition(-50, 0, 25)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        self._renderer.SetActiveCamera(camera)

        # Window
        self._render_window.SetSize(640, 480)
        self._render_window.SetPosition(0, 0)
