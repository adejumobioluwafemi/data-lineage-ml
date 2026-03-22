"""
LineageGraph — builds and renders a directed lineage graph
from logged steps using NetworkX + Plotly.
"""

import sys
from typing import Optional
from ..storage.sqlite_store import LineageStore


def _check_viz_deps():
    """Raise a helpful error if plotly or networkx are missing."""
    missing = []
    for pkg in ("plotly", "networkx"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        raise ImportError(
            f"Missing visualization dependencies: {', '.join(missing)}\n\n"
            f"Install them with:\n"
            f"    pip install {' '.join(missing)}\n\n"
            f"Or install the full extras:\n"
            f"    pip install \"datalineageml[viz]\"\n\n"
            f"Active Python: {sys.executable}\n"
            f"If plotly is installed but not found, make sure you are running\n"
            f"this script with the same Python that has plotly installed:\n"
            f"    {sys.executable} examples/basic_pipeline.py"
        )


class LineageGraph:
    def __init__(self, store: LineageStore = None):
        self.store = store or LineageStore()

    def build(self):
        """Return a NetworkX DiGraph of step → step edges."""
        _check_viz_deps()
        import networkx as nx

        steps = self.store.get_steps()
        G = nx.DiGraph()

        prev = None
        for step in steps:
            node_id = f"{step['step_name']}\n{step['started_at'][:19]}"
            G.add_node(node_id, **{
                "status": step["status"],
                "duration_ms": step["duration_ms"],
                "output_hash": step["output_hash"],
            })
            if prev:
                G.add_edge(prev, node_id)
            prev = node_id
        return G

    def show(self, output_html: Optional[str] = None):
        """Render an interactive Plotly graph. Optionally save to HTML file."""
        _check_viz_deps()
        import plotly.graph_objects as go
        import networkx as nx

        G = self.build()

        if len(G.nodes()) == 0:
            print("  LineageGraph: no steps logged yet — nothing to render.")
            return

        pos = nx.spring_layout(G, seed=42)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = list(G.nodes())
        node_colors = [
            "#22c55e" if G.nodes[n].get("status") == "success" else "#ef4444"
            for n in G.nodes()
        ]
        hover_text = [
            f"{n}<br>status: {G.nodes[n].get('status')}<br>"
            f"duration: {G.nodes[n].get('duration_ms')}ms<br>"
            f"out_hash: {str(G.nodes[n].get('output_hash', ''))[:12]}..."
            for n in G.nodes()
        ]

        fig = go.Figure(data=[
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=1.5, color="#94a3b8"),
                hoverinfo="none",
            ),
            go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(
                    size=28, color=node_colors,
                    line=dict(width=2, color="white"),
                ),
                text=node_text,
                hovertext=hover_text,
                hoverinfo="text",
                textposition="top center",
                textfont=dict(size=11),
            ),
        ])
        fig.update_layout(
            title=dict(
                text="DataLineageML — Pipeline Lineage Graph",
                font=dict(size=16),
            ),
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="🟢 success  🔴 failed",
                    showarrow=False, x=0.01, y=0.01,
                    xref="paper", yref="paper",
                    font=dict(size=11, color="#64748b"),
                )
            ]
        )

        if output_html:
            fig.write_html(output_html)
            print(f"  Saved lineage graph → {output_html}")
        else:
            fig.show()
