import React from "react";
import PropTypes from "prop-types";
import { observer } from "mobx-react";
import EdgeModel from "../../../models/Edge";
import Item from "./Item";
import Data from "./Data";
import IconsBase from "../../../../nodegraph-base/views/Icons";
import Style from "./ItemNode.module.css";

class ItemEdge extends Item
{
	constructor( tProps )
	{
		// Inheritance
		super( tProps );
		
		// Events
		this._onWeight = ( tEvent ) => { this.props.model.weight = tEvent.target.value; };
	}
	
	render( tStyle = Style )
	{
		return super.render( tStyle );
	}
	
	renderBar( tStyle = Style )
	{
		const tempModel = this.props.model;
		
		return (
			<button className={ tStyle.toggle } onClick={ this._onOpen }>
				{ IconsBase.arrow }
				<span>{ tempModel._type.text + ": " + tempModel._source._node._id + " to " + tempModel._target._node._id }</span>
			</button>
		);
	}
	
	renderContent( tStyle = Style )
	{
		const tempModel = this.props.model;
		
		return (
			<React.Fragment>
				<div className={ tStyle.kvp }>
					<span>Weight</span>
					<input type="number" value={ tempModel.weight } onChange={ this._onWeight } disabled={ !this.props.isEditable }/>
				</div>
				<Data data={ tempModel.data } isEditable={ this.props.isEditable }/>
			</React.Fragment>
		);
	}
}

ItemEdge.propTypes = Object.assign(
	{
		model: PropTypes.instanceOf( EdgeModel ).isRequired
	},
	Item.propTypes
);

export default observer( ItemEdge );