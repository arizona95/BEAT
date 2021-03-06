import React from "react";
import PropTypes from "prop-types";
import { observer } from "mobx-react";
import GraphModel from "../../../models/Graph";
import ArrowsBase from "../../../../nodegraph-base/views/graph/edges/Arrows";

class Arrows extends ArrowsBase
{
	render() // TODO: clean repeated code
	{
		const tempTypes = this.props.graph._edgeTypes;
		
		return (
			<defs>
				{
					Object.keys( tempTypes ).map(
						( tKey ) =>
						(
							<marker key={ tKey } id={ "arrow-" + tKey } markerWidth="10" markerHeight="10" viewBox="-10 -5 10 10" orient="auto" fill={ tempTypes[ tKey ].stroke }>
								<path d="M 0 0 L -10 5 L -10 -5 z"/>
							</marker>
						)
					)
				}
			</defs>
		);
	}
}

export default observer( Arrows );

Arrows.propTypes = Object.assign(
	{},
	ArrowsBase.propTypes,
	{
		graph: PropTypes.instanceOf( GraphModel ).isRequired
	}
);