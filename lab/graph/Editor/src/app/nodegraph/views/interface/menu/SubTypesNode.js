import React from "react";
import PropTypes from "prop-types";
import { observer } from "mobx-react";
import GraphModel from "../../../models/Graph";
import TypeModel from "../../../models/TypeNode";
import SubTypes from "./SubTypes";
import ItemNodeType from "./ItemNodeType";

class SubTypesNode extends SubTypes
{

	onNew()
	{
		if ( this.state.newKey !== "" && this.props.graph._nodeTypes[ this.state.newKey ] === undefined )
		{
			this.props.graph.setNodeType( this.createType( this.state.newKey ) );
			this.setState( { newKey: "" } );
		}
	}
	
	createType( tName )
	{
		return new TypeModel( tName );
	}
	
	renderItems()
	{
	    const spaceTypeModel = new TypeModel("Space");
	    spaceTypeModel.fill="#4D52F5"
	    spaceTypeModel.radius=50
        this.props.graph.setNodeType(spaceTypeModel);

	    const reactionTypeModel = new TypeModel("Reaction");
	    reactionTypeModel.fill="#A39F9F"
	    reactionTypeModel.radius=24
        this.props.graph.setNodeType(reactionTypeModel);

        const elementTypeModel = new TypeModel("Element");
	    elementTypeModel.fill="#C4C70F"
	    elementTypeModel.radius=30
        this.props.graph.setNodeType(elementTypeModel);

        const externelTypeModel = new TypeModel("External");
	    externelTypeModel.fill="#000000"
	    externelTypeModel.radius=30
        this.props.graph.setNodeType(externelTypeModel);

		return (
			<React.Fragment>
				{
					Object.keys( this.props.graph._nodeTypes ).map(
						( tKey ) =>
						(
							<ItemNodeType key={ tKey } graph={ this.props.graph } model={ this.props.graph._nodeTypes[ tKey ] } isEditable={ this.props.isEditable }/>
						)
					)
				}
			</React.Fragment>
		);
	}
}

SubTypesNode.propTypes = Object.assign(
	{
		graph: PropTypes.instanceOf( GraphModel ).isRequired
	},
	SubTypes.propTypes
);

SubTypesNode.defaultProps = Object.assign(
	{
		title: "node types"
	},
	SubTypes.defaultProps
);

export default observer( SubTypesNode );