import React from "react";
import PropTypes from "prop-types";
import { observer } from "mobx-react";
import Vector2D from "../../../../core/Vector2D";
import GraphModel from "../../../models/Graph";
import TypeModel from "../../../models/TypeNode";
import Item from "./Item";
import IconsBase from "../../../../nodegraph-base/views/Icons";
import Icons from "../../Icons";
import Style from "./ItemNodeType.module.css";

class ItemNodeType extends Item
{
	constructor( tProps )
	{
		// Inheritance
		super( tProps );
		
		// Events
		this._onSelect = () => { this.onSelect(); };
		this._onAdd = () => { this.onAdd(); };
		this._onDelete = () => { this.onDelete(); };
		this._onVisible = () => { this.props.model.isVisible = !this.props.model.isVisible; };
		this._onRadius = ( tEvent ) => { this.props.model.radius = parseInt( tEvent.target.value ); };
		this._onFill = ( tEvent ) => { this.props.model.fill = tEvent.target.value; };
	}
	
	onSelect()
	{
		const tempGraph = this.props.graph;
		for ( let tempID in tempGraph._nodes )
		{
			let tempNode = tempGraph._nodes[ tempID ];
			if ( tempNode._type === this.props.model )
			{
				tempGraph.setSelectedNode( tempNode );
			}
		}
	}
	
	onAdd()
	{
		const tempGraph = this.props.graph;
		const tempNode = new this.props.model._modelClass( tempGraph, this.props.model );
		tempNode.position = new Vector2D( window.screen.width * 0.5, window.screen.height * 0.5 ).scale( 1 / tempGraph.zoom ).subtract( tempGraph.position );

		if( tempNode._type._name == "Reaction"){
		    tempNode.data.k=1;
		}else if( tempNode._type._name == "Space"){
		    tempNode.data.V=1
		    tempNode.data.explanation=""
		}else if( tempNode._type._name == "Element"){
		    tempNode.data.q=0;
		    tempNode.data.m=1;
		    tempNode.data.explanation=""
		}else if( tempNode._type._name == "External"){
		    tempNode.data.x_h=1;
		    tempNode.data.explanation=""
		}else {
		    tempNode.data.A=1;
		    tempNode.data.C_0=1;
		    tempNode.data.c=1;
		    tempNode.data.input="";
		}

		tempGraph.setSelectedNode( tempNode );
		tempGraph.setNode( tempNode );
		console.log("here!@1",tempNode);
		
		return tempNode;
	}
	
	onDelete()
	{
		this.props.graph.removeNodeType( this.props.model );
	}
	
	renderBar( tStyle = Style )
	{
		const tempModel = this.props.model;
		
		// Class
		var tempVisibleClass = `${ tStyle.button }`;
		if ( !tempModel.isVisible )
		{
			tempVisibleClass += ` ${ tStyle.invisible }`;
		}
		
		// Render
		return (
			<React.Fragment>
				<button className={ tStyle.toggle } onClick={ this._onOpen }>
					{ IconsBase.arrow }
					<div className={ tStyle.circle } style={ { backgroundColor: tempModel.fill } }/>
					<span>{ tempModel._name }</span>
				</button>
				<div>
					<button className={ tempVisibleClass } onClick={ this._onVisible }>
						{ Icons.visible }
					</button>
					<button className={ tStyle.button } onClick={ this._onSelect }>
						{ Icons.select }
					</button>
					{
						this.props.isEditable &&
							<React.Fragment>
								<button className={ tStyle.button } onClick={ this._onAdd }>
									{ Icons.addNode }
								</button>
								<button className={ tStyle.button } onClick={ this._onDelete }>
									{ Icons.delete }
								</button>
							</React.Fragment>
					}
				</div>
			</React.Fragment>
		);
	}
	
	renderContent( tStyle = Style )
	{
		const tempModel = this.props.model;
		
		return (
			<div className={ tStyle.kvp }>
				<span>Radius</span>
				<input type="number" value={ tempModel.radius } onChange={ this._onRadius } disabled={ !this.props.isEditable }/>
				<span>Fill</span>
				<input type="color" value={ tempModel.fill } onChange={ this._onFill } disabled={ !this.props.isEditable }/>
			</div>
		);
	}
}

ItemNodeType.propTypes = Object.assign(
	{
		model: PropTypes.instanceOf( TypeModel ).isRequired,
		graph: PropTypes.instanceOf( GraphModel ).isRequired
	},
	Item.propTypes
);

export default observer( ItemNodeType );