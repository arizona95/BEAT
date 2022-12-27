import React from "react";
import PropTypes from "prop-types";
import { observer } from "mobx-react";
import GraphModel from "../../../models/Graph";
import TypeModel from "../../../models/TypeEdge";
import SubTypes from "./SubTypes";
import ItemEdgeType from "./ItemEdgeType";

class SubTypesEdge extends SubTypes
{
	onNew()
	{
		if ( this.state.newKey !== "" && this.props.graph._edgeTypes[ this.state.newKey ] === undefined )
		{
			this.props.graph.setEdgeType( this.createType( this.state.newKey ) );
			this.setState( { newKey: "" } );
		}
	}
	
	createType( tName )
	{
		return new TypeModel( tName );
	}
	
	renderItems()
	{
	    const chemicalReactionTypeModel = new TypeModel("ChemicalReaction");
	    chemicalReactionTypeModel.stroke="#2A3563"
	    chemicalReactionTypeModel.text="r"
        this.props.graph.setEdgeType(chemicalReactionTypeModel);

        const spaceElementTypeModel = new TypeModel("SpaceElement");
	    spaceElementTypeModel.stroke="#7D90CA"
	    spaceElementTypeModel.text="s"
        this.props.graph.setEdgeType(spaceElementTypeModel);

        const spaceNeighborTypeModel = new TypeModel("SpaceNeighbor");
	    spaceNeighborTypeModel.stroke="#4D52F5"
	    spaceNeighborTypeModel.text="n"
        this.props.graph.setEdgeType(spaceNeighborTypeModel);

        const consistTypeModel = new TypeModel("Consist");
	    consistTypeModel.stroke="#859532"
	    consistTypeModel.text="c"
        this.props.graph.setEdgeType(consistTypeModel);

        const hamiltonianDiffusionTypeModel = new TypeModel("HamiltonianDiffusion");
	    hamiltonianDiffusionTypeModel.stroke="#D6104F"
	    hamiltonianDiffusionTypeModel.text="f"
        this.props.graph.setEdgeType(hamiltonianDiffusionTypeModel);

        const externalDiffusionTypeModel = new TypeModel("ExternalDiffusion");
	    externalDiffusionTypeModel.stroke="#000000"
	    externalDiffusionTypeModel.text="h"
        this.props.graph.setEdgeType(externalDiffusionTypeModel);

		return (
			<React.Fragment>
				{
					Object.keys( this.props.graph._edgeTypes ).map(
						( tKey ) =>
						(
							<ItemEdgeType key={ tKey } graph={ this.props.graph } model={ this.props.graph._edgeTypes[ tKey ] } isEditable={ this.props.isEditable }/>
						)
					)
				}
			</React.Fragment>
		);
	}
}

SubTypesEdge.propTypes = Object.assign(
	{
		graph: PropTypes.instanceOf( GraphModel ).isRequired
	},
	SubTypes.propTypes
);

SubTypesEdge.defaultProps = Object.assign(
	{
		title: "edge types"
	},
	SubTypes.defaultProps
);

export default observer( SubTypesEdge );